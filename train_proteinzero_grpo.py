#!/usr/bin/env python
"""
GRPO fine-tuning for structure-conditioned protein sequence design.

The policy is trained as a LoRA adapter with TRL's GRPOTrainer. Rollouts are
restricted to the canonical 20-amino-acid action space and the target backbone
length. The default multi-objective reward is the fixed weighted sum described
in Section 3.1.2 of the paper.

Examples:
    python train_proteinzero_grpo.py --model_name ./MPNN-ProGen2-xlarge-CATH42
    torchrun --nproc_per_node=4 train_proteinzero_grpo.py --model_name ./MPNN-ProGen2-xlarge-CATH42
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import copy
import math
import hashlib
import json
import csv
import random
import re
import pickle
import tempfile
import subprocess
import logging
from collections import Counter
from pathlib import Path
from importlib.metadata import version as package_version
from functools import lru_cache
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tqdm import tqdm
from datasets import Dataset

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    HfArgumentParser,
    LogitsProcessor,
    LogitsProcessorList,
)
from peft import LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig
from trl.models import unwrap_model_for_generation

import esm
from esm.esmfold.v1.misc import output_to_pdb as _esm_output_to_pdb

# Load the vendored ProteinMPNN implementation used for stability scoring.
_mpnn_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ProteinMPNN")
sys.path.append(_mpnn_path)
from protein_mpnn_utils import _scores as mpnn_scores, tied_featurize, parse_PDB, ProteinMPNN

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Shared state initialized by main().
ESMFOLD_MODEL = None          # ESMFold model for folding/pLDDT/scRMSD
TOKENIZER = None              # ProGen2 tokenizer
DEVICE = None                 # Current CUDA device
PDB_TRAIN_DIR = None          # Path to native PDB structures
TRAIN_LIST = None             # List of {structure_id, native_seq, ...}
VAL_LIST = None               # List of {structure_id, native_seq, ...} (validation)

# Frozen ProteinMPNN scorer state.
mpnn_model          = None    # ProteinMPNN model instance
mpnn_feats_cache    = {}      # sid → (feats_cond, feats_uncond)
native_seqs         = {}      # sid → cleaned WT sequence string
mpnn_gen            = None    # torch.Generator, seeded from the run's --seed
MPNN_ALPHABET       = 'ACDEFGHIKLMNPQRSTVWYX'
MPNN_ALPHABET_DICT  = dict(zip(MPNN_ALPHABET, range(21)))

# Immutable action space shared by rollout, replay, reference, and validation.
PROTEIN_ACTION_SPACE = None
SID_TO_TARGET_INFO = {}       # {structure_id: {"target_len": int}}

# Training diagnostic state.

TOP_RATIO = 1.0               # Fraction retained per structure for diagnostic statistics; 1.0 retains all sequences.
REWARD_WEIGHTS_GLOBAL = None   # Set in main(), non-zero training objective weights

_TRAIN_METRICS_ACC = {
    "rr_sum": 0.0,
    "tm_sum": 0.0,
    "scrmsd_sum": 0.0,
    "ddg_sum": 0.0,
    "reward_sum": 0.0,
    "plddt_sum": 0.0,
    "cnt": 0,
    "lt2_cnt": 0,
    "div_sum": 0.0,
    "div_struct_cnt": 0,
}


# Command-line arguments
@dataclass
class ProteinGRPOArguments:
    """Arguments specific to the protein design GRPO training."""

    model_name: str = field(
        default="./MPNN-ProGen2-xlarge-CATH42",
        metadata={"help": "Path to the base ProGen2 model"},
    )
    cath_version: str = field(
        default="4.3",
        metadata={"help": "CATH dataset version: 4.3. Sets PDB directories, "
                          "train/validation CSV files, and the structure embedding "
                          "prefix when those are not set."},
    )
    pdb_train_dir: str = field(
        default=None,
        metadata={"help": "Directory containing native PDB files for training (default: from --cath_version)."},
    )
    pdb_val_dir: str = field(
        default=None,
        metadata={"help": "Directory containing native PDB files for validation (default: from --cath_version)."},
    )
    train_csv: str = field(
        default=None,
        metadata={"help": "CSV file with training data (default: from --cath_version, e.g. cath43/train_data_le100.csv)."},
    )
    val_csv: str = field(
        default=None,
        metadata={"help": "CSV file with validation data (default: from --cath_version)."},
    )
    structure_emb_path_prefix: str = field(
        default=None,
        metadata={"help": "Prefix dir for structure embedding .pyd files (default: from --cath_version)."},
    )
    # LoRA configuration
    lora_rank: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})

    # Reward weights (passed to GRPOTrainer.reward_weights)
    reward_weight_tm: float = field(default=1.0, metadata={"help": "Weight for TM-score reward"})
    reward_weight_ddg: float = field(
        default=1.0,
        metadata={"help": "Weight for physical ΔΔG reward (lower is better)"},
    )
    reward_weight_plddt: float = field(default=0.5, metadata={"help": "Weight for pLDDT reward"})
    reward_weight_recovery: float = field(default=0.0, metadata={"help": "Weight for recovery reward"})
    reward_weight_length: float = field(default=2.0, metadata={"help": "Weight for length-match reward"})
    # The gen_* parameters configure GRPOConfig; eval_* parameters are used only
    # by evaluate_all_metrics and the validation callback.
    gen_temperature: float = field(default=1.0, metadata={"help": "GRPO rollout temperature (→ GRPOConfig.temperature)"})
    gen_top_p: float = field(default=1.0, metadata={"help": "GRPO rollout top_p (→ GRPOConfig.top_p)"})
    eval_temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for validation / evaluate_all_metrics only (not GRPO rollouts)"},
    )
    eval_top_p: float = field(
        default=1.0,
        metadata={"help": "top_p for validation / evaluate_all_metrics only"},
    )
    val_num_generations: int = field(
        default=1,
        metadata={"help": "Number of validation sequences generated per backbone globally; independent of GRPO rollout num_generations."},
    )
    kl_beta: float = field(default=0.1, metadata={"help": "KL penalty coefficient (→ GRPOConfig.beta)"})

    # These fields are inherited from GRPOConfig; redeclaration would create
    # conflicting command-line options.

    # Diversity penalty (embedding-space, applied in loss)
    alpha_diversity: float = field(default=0.05, metadata={
        "help": "Embedding-space diversity penalty coefficient (0 to disable)"
    })

    # Sequence-length constraint.
    force_exact_length: bool = field(
        default=True,
        metadata={"help": "Constrain generated sequences to the native backbone length. "
                          "Default: True."},
    )

    # resume_from_checkpoint is inherited from TrainingArguments; redeclaration
    # would create conflicting command-line options.

    # Misc
    validate_after_training: bool = field(default=True, metadata={"help": "Run validation after training"})
    validate_every_steps: int = field(
        default=0,
        metadata={"help": "Validate every N training steps when N > 0; 0 disables step-based validation."},
    )


# General utilities

def seq_diversity(seq_list):
    """
    Return the mean fraction of differing positions over all unordered pairs
    of equal-length sequences. Return 0.0 for fewer than two sequences.
    """
    n = len(seq_list)
    if n < 2:
        return 0.0
    L = len(seq_list[0])
    diffs = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            diffs += sum(a != b for a, b in zip(seq_list[i], seq_list[j])) / L
            pairs += 1
    if pairs == 0:
        return 0.0
    return diffs / pairs


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path, block_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


CANONICAL_AAS = tuple("ACDEFGHIKLMNPQRSTVWY")
EXPECTED_AA_TOKEN_IDS = (5, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                         16, 17, 19, 20, 21, 22, 23, 25, 26, 28)
AA_REGEX = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")
EXPECTED_POLICY_VERSIONS = {
    "torch": "2.10.0", "transformers": "5.1.0",
    "accelerate": "1.12.0", "trl": "0.27.2", "peft": "0.18.1",
}


@dataclass(frozen=True)
class ProteinActionSpace:
    vocab_size: int
    aa_token_ids: tuple
    sentinel1_id: int
    sentinel2_id: int
    special_eos_id: int
    pad_id: int

    @property
    def aa_token_id_set(self):
        return frozenset(self.aa_token_ids)


def _one_token(tokenizer, text, label):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"{label} must be exactly one token, got {ids}")
    return int(ids[0])


def build_protein_action_space(tokenizer, model_config):
    """Freeze A20/control IDs; model config, not raw tokenizer metadata, owns special EOS."""
    special_eos_id = getattr(model_config, "eos_token_id", None)
    if special_eos_id != 2:
        raise RuntimeError(f"Model config special EOS must be 2, got {special_eos_id}")
    raw_tokenizer_eos = tokenizer.eos_token_id
    if raw_tokenizer_eos not in (None, special_eos_id):
        raise RuntimeError(
            f"Tokenizer EOS {raw_tokenizer_eos} conflicts with model EOS {special_eos_id}"
        )
    # The tokenizer does not declare EOS; the model configuration is the
    # authoritative source for the runtime contract.
    tokenizer.eos_token_id = int(special_eos_id)
    aa_ids = tuple(_one_token(tokenizer, aa, f"AA {aa}") for aa in CANONICAL_AAS)
    spec = ProteinActionSpace(
        vocab_size=len(tokenizer), aa_token_ids=aa_ids,
        sentinel1_id=_one_token(tokenizer, "1", "sentinel 1"),
        sentinel2_id=_one_token(tokenizer, "2", "sentinel 2"),
        special_eos_id=int(special_eos_id), pad_id=int(tokenizer.pad_token_id),
    )
    if (spec.vocab_size != 30 or len(set(spec.aa_token_ids)) != 20
            or spec.aa_token_ids != EXPECTED_AA_TOKEN_IDS):
        raise RuntimeError(f"Invalid protein vocabulary contract: {spec}")
    if (spec.pad_id, spec.special_eos_id, spec.sentinel1_id, spec.sentinel2_id) != (0, 2, 3, 4):
        raise RuntimeError(f"Invalid control-token contract: {spec}")
    if set(spec.aa_token_ids) & {spec.pad_id, spec.special_eos_id, spec.sentinel1_id, spec.sentinel2_id}:
        raise RuntimeError("A20 IDs overlap control/special tokens")
    logger.info(f"Protein action space: {spec}")
    return spec


def protein_action_logps_and_entropies(logits, token_ids, action_space, temperature, compute_entropy=False):
    """A20-normalized log-probs; non-actions return exact zero."""
    if logits.shape[:-1] != token_ids.shape or logits.size(-1) != action_space.vocab_size:
        raise ValueError(f"Bad protein logits/token shape: {logits.shape}, {token_ids.shape}")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError(f"Invalid temperature {temperature}")
    aa_ids = torch.tensor(action_space.aa_token_ids, device=logits.device)
    aa_logps = F.log_softmax(
        logits.index_select(-1, aa_ids).float() / float(temperature),
        dim=-1, dtype=torch.float32,
    )
    lookup = torch.full((action_space.vocab_size,), -1, dtype=torch.long, device=logits.device)
    lookup[aa_ids] = torch.arange(len(aa_ids), device=logits.device)
    action_index = lookup[token_ids]
    is_action = action_index.ge(0)
    selected = aa_logps.gather(-1, action_index.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    logps = torch.where(is_action, selected, torch.zeros_like(selected))
    entropies = None
    if compute_entropy:
        with torch.no_grad():
            finite_entropy = -(aa_logps.exp() * aa_logps).sum(-1)
            entropies = torch.where(is_action, finite_entropy, torch.zeros_like(finite_entropy))
    return logps, entropies


def configure_strict_lm_policy_runtime(config, protein_args):
    if sys.version_info[:3] != (3, 10, 19) or torch.__version__ != "2.10.0+cu128":
        logger.warning(
            "Runtime differs from the verified Python/PyTorch versions: "
            f"verified=3.10.19/2.10.0+cu128, "
            f"current={sys.version.split()[0]}/{torch.__version__}"
        )
    for package, expected in EXPECTED_POLICY_VERSIONS.items():
        actual = package_version(package)
        if actual != expected:
            logger.warning(
                f"Runtime differs from the verified {package} version: "
                f"verified={expected}, current={actual}"
            )
    config.disable_dropout = True
    config.fp16 = config.bf16 = False
    config.fp16_full_eval = config.bf16_full_eval = False
    config.tf32 = False
    # Transformers 5.1 transformers/training_args.py:1551-1555 derives this
    # non-dataclass attribute; transformers/trainer.py:5066 passes it to Accelerator.
    # The runtime value must be derived from the pre-parse environment and the
    # fp16/bf16 configuration before this validation.
    mixed_precision = getattr(config, "mixed_precision", None)
    if mixed_precision != "no":
        raise RuntimeError(
            "TrainingArguments mixed_precision contract failed: "
            f"expected='no', actual={mixed_precision!r}, "
            f"type={type(mixed_precision).__name__}"
        )
    missing = object()

    def config_value(name, default=missing):
        return getattr(config, name, default)

    def config_actual(value):
        if value is missing:
            return "actual=<MISSING>, type=MISSING"
        return f"actual={value!r}, type={type(value).__name__}"

    top_p = config_value("top_p")
    top_k = config_value("top_k")
    min_p = config_value("min_p", None)
    repetition_penalty = config_value("repetition_penalty")
    use_vllm = config_value("use_vllm", False)
    loss_type = config_value("loss_type")
    num_iterations = config_value("num_iterations")
    importance_sampling_level = config_value("importance_sampling_level")
    violations = []
    checks = (
        (protein_args.force_exact_length, "force_exact_length must be True"),
        (top_p == 1.0, f"top_p expected=1.0, {config_actual(top_p)}"),
        (top_k == 0, f"top_k expected=0, {config_actual(top_k)}"),
        (min_p is None, f"min_p expected=None, {config_actual(min_p)}"),
        (repetition_penalty == 1.0,
         f"repetition_penalty expected=1.0, {config_actual(repetition_penalty)}"),
        (not bool(use_vllm), f"use_vllm expected=False, {config_actual(use_vllm)}"),
        (not getattr(config, "use_transformers_paged", False), "paged generation must be False"),
        (not getattr(config, "use_liger_kernel", False), "Liger must be False"),
        (not getattr(config, "torch_compile", False), "torch_compile must be False"),
        (not getattr(config, "generation_kwargs", None), "generation_kwargs must be empty"),
        (getattr(config, "cache_implementation", None) in (None, "dynamic"),
         f"cache_implementation={getattr(config, 'cache_implementation', None)}"),
        (loss_type == "grpo", f"loss_type expected='grpo', {config_actual(loss_type)}"),
        (num_iterations == 1,
         f"num_iterations expected=1, {config_actual(num_iterations)}"),
        (importance_sampling_level == "token",
         "importance_sampling_level expected='token', "
         f"{config_actual(importance_sampling_level)}"),
    )
    violations.extend(message for passed, message in checks if not passed)
    if violations:
        raise RuntimeError("Unsupported LM policy config: " + "; ".join(violations))
    # TRL 0.27.2 grpo_config.py:888-904 derives steps_per_generation in
    # GRPOConfig.__post_init__; grpo_trainer.py:1914-1915 uses this exact
    # alignment condition to decide whether it can skip the old-policy forward.
    steps_per_generation = config_value("steps_per_generation")
    gradient_accumulation_steps = config_value("gradient_accumulation_steps")
    alignment_values = {
        "steps_per_generation": steps_per_generation,
        "num_iterations": num_iterations,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    invalid_alignment = {
        name: value for name, value in alignment_values.items()
        if value is missing or isinstance(value, bool)
        or not isinstance(value, int) or value <= 0
    }
    if invalid_alignment:
        details = "; ".join(
            f"{name} expected=positive int, {config_actual(value)}"
            for name, value in invalid_alignment.items()
        )
        raise RuntimeError("Unsupported LM policy config: " + details)
    generate_every = steps_per_generation * num_iterations
    if gradient_accumulation_steps % generate_every != 0:
        raise RuntimeError(
            "Unsupported LM policy config: gradient_accumulation_steps="
            f"{gradient_accumulation_steps} must be divisible by "
            "steps_per_generation*num_iterations="
            f"{steps_per_generation}*{num_iterations}={generate_every}"
        )
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "").strip() == "1":
        raise RuntimeError("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 conflicts with strict FP32")
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "fp32_precision"):
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def assert_ieee_fp32_runtime():
    if hasattr(torch.backends, "fp32_precision"):
        values = (torch.backends.fp32_precision, torch.backends.cuda.matmul.fp32_precision,
                  torch.backends.cudnn.fp32_precision)
        if any(value != "ieee" for value in values):
            raise RuntimeError(f"Non-IEEE FP32 backend state: {values}")
    elif torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 is enabled")


def parse_structure_prompt(prompt):
    if not isinstance(prompt, str):
        raise RuntimeError(f"Prompt must be str, got {type(prompt).__name__}")
    parts = prompt.rsplit("|", 1)
    if len(parts) != 2 or not parts[0].strip() or parts[1] != "1":
        raise RuntimeError(f"Malformed protein prompt {prompt!r}; expected '<sid>|1'")
    return parts[0].strip()


def clean(completion_text: str) -> str:
    """
    Extract a canonical amino-acid sequence from a full model response
    ("{sid}|1MKVL...2") or completion-only text ("MKVL...2").
    """
    text = completion_text.strip()
    # Retain tokens following sentinel "1".
    bos = text.find("1")
    if bos >= 0:
        text = text[bos + 1:]
    # Retain tokens preceding sentinel "2".
    eos = text.find("2")
    if eos >= 0:
        text = text[:eos]
    return AA_REGEX.sub("", text).upper()


class ForceExactLength(LogitsProcessor):
    """
    Forces the generated sequence to have EXACTLY `target_len` residues,
    AND ensures only valid amino acid tokens are generated.

    - Before target_len residues:
        - Block the sentinel token "2"
        - Block all non-amino-acid tokens
    - At exactly target_len residues: force the sentinel token "2"
    """
    def __init__(self, action_space=None, target_len=None, prefix_len=None,
                 tokenizer=None, aa_token_ids=None, sentinel_id=None):
        # Evaluation entry points may supply these keyword arguments; every value
        # must match the immutable global action-space specification.
        if isinstance(action_space, ProteinActionSpace):
            spec = action_space
        else:
            if action_space is not None:
                legacy_sentinel = int(action_space)
                if sentinel_id is not None and int(sentinel_id) != legacy_sentinel:
                    raise RuntimeError("Conflicting legacy sentinel IDs")
                sentinel_id = legacy_sentinel
            spec = PROTEIN_ACTION_SPACE
            if spec is None:
                raise RuntimeError(
                    "ProteinActionSpace has not been initialized from tokenizer + model config"
                )
        if sentinel_id is not None and int(sentinel_id) != spec.sentinel2_id:
            raise RuntimeError("Legacy sentinel ID disagrees with ProteinActionSpace")
        if aa_token_ids is not None and set(aa_token_ids) != spec.aa_token_id_set:
            raise RuntimeError("Legacy AA IDs disagree with ProteinActionSpace")
        self.action_space = spec
        self.sentinel_id = spec.sentinel2_id
        self.target_len = int(target_len)
        self._prefix_len = prefix_len
        self.aa_token_ids = spec.aa_token_id_set
        if self.target_len <= 0:
            raise ValueError(f"target_len must be positive, got {self.target_len}")

    def __call__(self, input_ids, scores):
        # Infer the prefix length from the first generation call.
        if self._prefix_len is None:
            self._prefix_len = input_ids.shape[1]

        cur_len = input_ids.shape[1]
        n_res   = cur_len - self._prefix_len       # residues emitted so far

        if n_res < self.target_len:
            # Mask the terminal sentinel and non-amino-acid tokens before the target length.
            scores[:, self.sentinel_id] = -float('inf')

            # Mask every token outside the amino-acid whitelist.
            if self.aa_token_ids:
                mask = torch.ones_like(scores) * float('-inf')
                for aa_id in self.aa_token_ids:
                    mask[:, aa_id] = 0.0
                scores = scores + mask

        else:
            # Continue forcing the sentinel after an unexpected overshoot so
            # that generation terminates through EOS handling.
            scores[:, :] = -float('inf')
            scores[:, self.sentinel_id] = 0.0

        return scores


def partial_recovery(seq: str, native_seq: str) -> float:
    """Compute position-wise recovery rate between seq and native_seq."""
    if not seq or not native_seq:
        return 0.0
    min_len = min(len(seq), len(native_seq))
    if min_len == 0:
        return 0.0
    matches = sum(1 for a, b in zip(seq[:min_len], native_seq[:min_len]) if a == b)
    return matches / len(native_seq)


# ESMFold utilities

def init_esmfold(device: torch.device):
    """Initialize ESMFold model on the given device."""
    global ESMFOLD_MODEL
    logger.info(f"Loading ESMFold on {device}...")
    ESMFOLD_MODEL = esm.pretrained.esmfold_v1().eval()
    ESMFOLD_MODEL = ESMFOLD_MODEL.to(device)
    ESMFOLD_MODEL.set_chunk_size(64)
    logger.info("ESMFold ready.")


@lru_cache(maxsize=50_000)
def _unified_esmfold_cache(seq: str):
    """Single ESMFold forward pass — returns (pdb_str, mean_plddt).

    Both downstream consumers (esmfold_predict_pdb and fast_plddt) read from
    this cache, so each unique sequence triggers exactly one forward pass.
    """
    global ESMFOLD_MODEL
    if not seq or len(seq) > 1000:
        return "", 0.0
    try:
        with torch.no_grad():
            output = ESMFOLD_MODEL.float().infer(seq)
        pdb_str = _esm_output_to_pdb(output)[0]
        if len(pdb_str) < 200:
            pdb_str = ""
        mean_plddt = float(output["mean_plddt"])
        return pdb_str, mean_plddt
    except Exception as e:
        logger.warning(f"ESMFold failed for seq length {len(seq)}: {e}")
        return "", 0.0


def esmfold_predict_pdb(seq: str) -> str:
    """Fold a sequence with ESMFold and return PDB string (via unified cache)."""
    return _unified_esmfold_cache(seq)[0]


# Structural metrics

_SC_RMSD_CACHE = {}   # key = (seq, sid, pdb_dir)  →  float RMSD


def compute_tm_score_with_esmfold(
    seq: str,
    structure_id: str,
    tm_align_path: str = "TMalign",
    pdb_dir: str = "./pdbs/train",
) -> float:
    """
    Fold *seq* with ESMFold, TM-align to native backbone, return TM-score.
    The corresponding scRMSD is cached.
    """
    global _SC_RMSD_CACHE

    native_pdb = os.path.join(pdb_dir, f"{structure_id}.pdb")
    if not os.path.isfile(native_pdb):
        return 0.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            pdb_str = esmfold_predict_pdb(seq)
            if not pdb_str:
                return 0.0

            pred_pdb = os.path.join(tmp_dir, "pred.pdb")
            with open(pred_pdb, "w") as f:
                f.write(pdb_str)

            result = subprocess.run(
                [tm_align_path, pred_pdb, native_pdb],
                capture_output=True, text=True, check=True,
            )

            tm_score = 0.0
            rmsd_val = 0.0

            for line in result.stdout.splitlines():
                line = line.strip()
                # First TM-score line in output
                m_tm = re.search(r"TM-score\s*=\s*([0-9.]+)", line)
                if m_tm and tm_score == 0.0:
                    tm_score = float(m_tm.group(1))
                # Any RMSD line
                m_rmsd = re.search(r"RMSD[^=]*=\s*([0-9.]+)", line)
                if m_rmsd:
                    rmsd_val = float(m_rmsd.group(1))

            _SC_RMSD_CACHE[(seq, structure_id, pdb_dir)] = rmsd_val
            return tm_score

        except Exception:
            _SC_RMSD_CACHE[(seq, structure_id, pdb_dir)] = 0.0
            return 0.0


@lru_cache(maxsize=50_000)
def fast_tm_score(seq: str, structure_id: str, pdb_dir: str = None) -> float:
    if pdb_dir is None:
        pdb_dir = PDB_TRAIN_DIR
    return compute_tm_score_with_esmfold(seq, structure_id, pdb_dir=pdb_dir)


@lru_cache(maxsize=50_000)
def fast_plddt(seq: str) -> float:
    """Return mean pLDDT for a sequence (via unified ESMFold cache)."""
    try:
        return _unified_esmfold_cache(seq)[1]
    except Exception:
        return 0.0


def fast_sc_rmsd(seq: str, structure_id: str, pdb_dir: str = None) -> float:
    """Compute scRMSD (Cα RMSD from TM-align)."""
    if pdb_dir is None:
        pdb_dir = PDB_TRAIN_DIR
    k = (seq, structure_id, pdb_dir)
    if k not in _SC_RMSD_CACHE:
        _ = fast_tm_score(seq, structure_id, pdb_dir=pdb_dir)
    return _SC_RMSD_CACHE.get(k, 0.0)


# ProteinMPNN ΔΔG estimator

KBT = 0.593  # kcal mol⁻¹ at 298 K


def load_mpnn_model(device, seed, dropout=0.0):
    """Load ProteinMPNN v_48_020 and return (model, generator).

    The seed is required with no silent fallback, dropout is explicit, and
    augment_eps is asserted. A single user-facing --seed controls all
    randomness including the ddG scorer's random decoding order.
    """
    weights_path = os.path.join(_mpnn_path, "vanilla_model_weights", "v_48_020.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"ProteinMPNN weights not found: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device)
    model = ProteinMPNN(
        ca_only=False, num_letters=21,
        node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3,
        # The root protein_mpnn_utils.py does not gate augment_eps on
        # self.training, so model.eval() cannot disable coordinate noise.
        augment_eps=0.0,
        dropout=dropout,
        k_neighbors=ckpt['num_edges'],
    )
    assert model.features.augment_eps == 0.0, (
        "ProteinMPNN coordinate augmentation must remain disabled"
    )
    assert all(
        module.p == dropout
        for module in model.modules()
        if isinstance(module, torch.nn.Dropout)
    ), f"ProteinMPNN dropout does not match requested value {dropout}"
    model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return model, gen


def prepare_mpnn_feats(pdb_path, device):
    """
    Parse a PDB file and return (feats_cond, feats_uncond).
    feats_uncond has all coordinates zeroed (unconditional prior).
    Returns (None, None) on failure.
    """
    try:
        pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
    except Exception:
        return None, None
    if not pdb_dict_list:
        return None, None

    pdb_dict_zero = copy.deepcopy(pdb_dict_list)
    for chain_key in list(pdb_dict_zero[0].keys()):
        if chain_key.startswith("coords_chain_"):
            coords_dict = pdb_dict_zero[0][chain_key]
            for k in coords_dict:
                arr = np.array(coords_dict[k])
                arr[:] = 0.0
                coords_dict[k] = arr.tolist()

    def _featurize(dlist):
        (_, X, _, mask, _,
         chain_M, chain_encoding,
         _, _, _, _,
         chain_M_pos, _,
         residue_idx,
         _, _, _, _, _, _, _) = tied_featurize(
            [dlist[0]], device, chain_dict=None,
            fixed_position_dict=None, omit_AA_dict=None,
            tied_positions_dict=None, pssm_dict=None,
            bias_by_res_dict=None, ca_only=False
        )
        return {"X": X, "mask": mask, "chain_M": chain_M,
                "chain_M_pos": chain_M_pos, "residue_idx": residue_idx,
                "chain_encoding": chain_encoding}

    try:
        feats_cond = _featurize(pdb_dict_list)
        feats_uncond = _featurize(pdb_dict_zero)
    except Exception:
        return None, None
    return feats_cond, feats_uncond


def mpnn_nll(seq_str, feats, mask_for_loss, randn):
    """
    Single ProteinMPNN forward pass → total NLL (nats).
    Uses externally provided mask_for_loss and randn for consistency.
    """
    L_eff = mask_for_loss.sum().item()
    if L_eff == 0:
        return None
    S = torch.tensor(
        [MPNN_ALPHABET_DICT.get(aa, 20) for aa in seq_str], device=DEVICE
    ).unsqueeze(0)
    if S.shape[1] != feats["X"].shape[1]:
        return None
    with torch.no_grad():
        log_probs, _ = mpnn_model(
            feats["X"], S, feats["mask"],
            feats["chain_M"] * feats["chain_M_pos"],
            feats["residue_idx"], feats["chain_encoding"], randn
        )
    score_mean = mpnn_scores(S, log_probs, mask_for_loss).mean()
    return score_mean.item() * L_eff


def fast_ddg(seq: str, structure_id: str) -> float:
    """
    Physical ΔΔG ≈ kBT * (dG_mut − dG_wt).

    where dG = NLL_cond − NLL_uncond  (ProteinMPNN, same randn for all 4).

    Negative values mean that the mutant has lower predicted energy than the
    wild type. This sign convention is used by training logs and validation.
    """
    if structure_id not in mpnn_feats_cache:
        return 0.0

    feats_cond, feats_uncond = mpnn_feats_cache[structure_id]
    wt_seq = native_seqs[structure_id]

    mask_for_loss = feats_cond["mask"] * feats_cond["chain_M"] * feats_cond["chain_M_pos"]
    randn = torch.randn(feats_cond["chain_M"].shape, device=DEVICE, generator=mpnn_gen)

    nll_cond_mut   = mpnn_nll(seq,    feats_cond,   mask_for_loss, randn)
    nll_uncond_mut = mpnn_nll(seq,    feats_uncond, mask_for_loss, randn)
    nll_cond_wt    = mpnn_nll(wt_seq, feats_cond,   mask_for_loss, randn)
    nll_uncond_wt  = mpnn_nll(wt_seq, feats_uncond, mask_for_loss, randn)

    if any(v is None for v in [nll_cond_mut, nll_uncond_mut,
                                nll_cond_wt, nll_uncond_wt]):
        return 0.0

    dG_mut = nll_cond_mut - nll_uncond_mut
    dG_wt  = nll_cond_wt  - nll_uncond_wt
    return KBT * (dG_mut - dG_wt)


# Training diagnostics

def _accumulate_train_metrics(completions, structure_ids, native_seqs, reward_values=None, ddg_values=None):
    """
    Compute ALL raw metrics for the current prompt group, select the top
    sequences per structure, and accumulate into _TRAIN_METRICS_ACC.

    Called once per reward-function batch (from multi_objective_reward).
    This does NOT affect reward values — it is purely for logging to txt files.

    If reward_values is provided (list of per-completion training rewards), it is
    included in the per-sequence print line alongside TM, pLDDT, etc.

    If ddg_values is provided, it contains physical per-completion ΔΔG values
    from the reward path (negative = more stable than wild type). They are
    reused directly instead of being recomputed, because recomputation would
    consume a different random decoding order.

    """
    global _TRAIN_METRICS_ACC

    # Group completions by structure_id (track original index for reward lookup)
    from collections import defaultdict
    groups = defaultdict(list)  # sid -> [(orig_idx, comp, nat)]
    for orig_idx, (comp, sid, nat) in enumerate(zip(completions, structure_ids, native_seqs)):
        groups[sid].append((orig_idx, comp, nat))

    for sid, items in groups.items():
        # Compute raw metrics for every sequence in this group
        raw_metrics = []  # (seq, rr, tm, scrmsd, plddt, ddg_raw, reward)
        for orig_idx, comp, nat in items:
            seq = clean(comp)
            if not seq or len(seq) < 10:
                continue
            try:
                rr = partial_recovery(seq, nat)
                tm_val = fast_tm_score(seq, sid)
                scrmsd = fast_sc_rmsd(seq, sid)
                plddt = fast_plddt(seq)
                if ddg_values is not None:
                    ddg_raw = ddg_values[orig_idx]
                else:
                    ddg_raw = fast_ddg(seq, sid)
                reward_val = reward_values[orig_idx] if reward_values is not None else None
                raw_metrics.append((seq, rr, tm_val, scrmsd, plddt, ddg_raw, reward_val))
            except Exception as e:
                logger.warning(f"_accumulate_train_metrics: metrics failed for {sid}: {e}")
                continue

        if not raw_metrics:
            continue

        # Gather diagnostics across ranks for rank-zero reporting.
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_gpu_metrics = [None] * dist.get_world_size()
            dist.all_gather_object(all_gpu_metrics, raw_metrics)
            if local_rank == 0:
                merged_metrics = []
                for gpu_metrics in all_gpu_metrics:
                    if gpu_metrics:
                        merged_metrics.extend(gpu_metrics)
                display_metrics = merged_metrics
            else:
                display_metrics = None
        else:
            display_metrics = raw_metrics

        if display_metrics is not None and len(display_metrics) > 0:
            per_seq_rewards = _compute_combined_reward_for_group(
                display_metrics, REWARD_WEIGHTS_GLOBAL
            )
            print(f"\nStructure {sid}: {len(display_metrics)} generated sequences =>")
            reward_ranks = {}
            reward_list = [(i, m[6]) for i, m in enumerate(display_metrics) if m[6] is not None]
            if reward_list:
                reward_list.sort(key=lambda x: x[1], reverse=True)
                for rank, (idx, _) in enumerate(reward_list, 1):
                    reward_ranks[idx] = rank
            for i, (seq, rr, tm_val, scrmsd, plddt, ddg_raw, reward_val) in enumerate(display_metrics):
                if reward_val is not None:
                    rank_str = (
                        f" (rank {reward_ranks[i]}/{len(reward_list)})"
                        if i in reward_ranks else ""
                    )
                    reward_str = f", reward={reward_val:.4f}{rank_str}"
                else:
                    reward_str = f", reward={per_seq_rewards[i]:.4f}"
                print(f"  rr={rr:.3f}, TM={tm_val:.3f}, scRMSD={scrmsd:.3f}, "
                      f"pLDDT={plddt:5.1f}, ΔΔGraw={ddg_raw:.3f}"
                      f"{reward_str} | {seq}")
            n_disp = len(display_metrics)
            print(f"  --- Averages (all {n_disp} seqs):")
            print(f"      partial_recovery = {sum(m[1] for m in display_metrics)/n_disp:.3f}")
            print(f"      TM-score         = {sum(m[2] for m in display_metrics)/n_disp:.3f}")
            print(f"      scRMSD           = {sum(m[3] for m in display_metrics)/n_disp:.3f}")
            print(f"      pLDDT            = {sum(m[4] for m in display_metrics)/n_disp:5.1f}")
            print(f"      ΔΔGraw           = {sum(m[5] for m in display_metrics)/n_disp:.3f}")
            print(f"      combined_reward  = {sum(per_seq_rewards)/n_disp:.3f}")
            valid_rewards = [m[6] for m in display_metrics if m[6] is not None]
            if valid_rewards:
                print(f"      avg_reward       = {sum(valid_rewards)/len(valid_rewards):.4f}")

        # Gather metrics across ranks for global ranking and accumulation.
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_gpu_metrics_acc = [None] * dist.get_world_size()
            dist.all_gather_object(all_gpu_metrics_acc, raw_metrics)
            global_metrics = []
            for gpu_metrics in all_gpu_metrics_acc:
                if gpu_metrics:
                    global_metrics.extend(gpu_metrics)
        else:
            global_metrics = raw_metrics

        # Rank the complete cross-rank set by combined reward.
        top_n = max(1, int(len(global_metrics) * TOP_RATIO))
        combined_scores = _compute_combined_reward_for_group(
            global_metrics, REWARD_WEIGHTS_GLOBAL
        )
        scored = list(zip(combined_scores, global_metrics))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_pairs = [m for _, m in scored[:top_n]]

        # Accumulate diagnostics once on rank zero.
        if local_rank == 0:
            for seq, rr, tm_val, scrmsd, plddt, ddg_raw, _reward in top_pairs:
                _TRAIN_METRICS_ACC["rr_sum"] += rr
                _TRAIN_METRICS_ACC["tm_sum"] += tm_val
                _TRAIN_METRICS_ACC["scrmsd_sum"] += scrmsd
                _TRAIN_METRICS_ACC["plddt_sum"] += plddt
                _TRAIN_METRICS_ACC["ddg_sum"] += ddg_raw
                _TRAIN_METRICS_ACC["cnt"] += 1
                if scrmsd < 2.0:
                    _TRAIN_METRICS_ACC["lt2_cnt"] += 1
            top_scores = [s for s, _ in scored[:top_n]]
            for s in top_scores:
                _TRAIN_METRICS_ACC["reward_sum"] += s

            top_seqs = [x[0] for x in top_pairs]
            div_val = seq_diversity(top_seqs)
            _TRAIN_METRICS_ACC["div_sum"] += div_val
            _TRAIN_METRICS_ACC["div_struct_cnt"] += 1


def _flush_train_metrics(step: int, output_dir: str):
    """
    Write accumulated training metrics to txt files (one value per logging interval)
    and reset the accumulator.
    """
    global _TRAIN_METRICS_ACC

    acc = _TRAIN_METRICS_ACC
    n = acc["cnt"]
    if n == 0:
        return

    avg_rr = acc["rr_sum"] / n
    avg_tm = acc["tm_sum"] / n
    avg_scrmsd = acc["scrmsd_sum"] / n
    avg_ddg = acc["ddg_sum"] / n
    avg_reward = acc["reward_sum"] / n
    avg_plddt = acc["plddt_sum"] / n
    frac_lt2 = acc["lt2_cnt"] / n
    avg_div = acc["div_sum"] / acc["div_struct_cnt"] if acc["div_struct_cnt"] > 0 else 0.0
    log_dir = output_dir
    os.makedirs(log_dir, exist_ok=True)

    txt_files = {
        "tm_score_train.txt": avg_tm,
        "partial_recovery_train.txt": avg_rr,
        "sc_rmsd_train.txt": avg_scrmsd,
        "ddg_score.txt": avg_ddg,
        "combined_reward.txt": avg_reward,
        "plddt_train.txt": avg_plddt,
        "scrmsd_lt2_train.txt": frac_lt2,
        "diversity_train.txt": avg_div,
    }

    for fname, value in txt_files.items():
        fpath = os.path.join(log_dir, fname)
        with open(fpath, "a") as f:
            f.write(f"{step}\t{value:.6f}\n")

    logger.info(f"[Step {step}] Training metrics (top-{int(TOP_RATIO*100)}% combined reward): "
                f"TM={avg_tm:.3f}, RR={avg_rr:.3f}, scRMSD={avg_scrmsd:.3f}, "
                f"pLDDT={avg_plddt:.1f}, ΔΔG={avg_ddg:.3f}, "
                f"frac_lt2={frac_lt2:.3f}, "
                f"div={avg_div:.3f} (n={n})")

    for key in _TRAIN_METRICS_ACC:
        _TRAIN_METRICS_ACC[key] = 0.0 if isinstance(_TRAIN_METRICS_ACC[key], float) else 0


# Reward functions

def _minmax_norm(val: float, lo: float, hi: float) -> float:
    """Min-max normalise *val* to [0, 1].  Returns 0.5 if lo == hi."""
    if hi - lo < 1e-12:
        return 0.5
    return (val - lo) / (hi - lo)


def _training_objective_norm(
    name: str,
    val: float,
    lo: float,
    hi: float,
    delta: float = 1e-8,
) -> float:
    """Normalize one training objective so that larger normalized values are better.

    Physical ΔΔG is the only lower-is-better objective. Using ``hi - val``
    preserves exact reward direction parity with the historical ``-ΔΔG``
    representation, including the training path's 0.0 convention when
    ``hi == lo``.
    """
    if name == "ddg":
        return (hi - val) / (hi - lo + delta)
    return (val - lo) / (hi - lo + delta)


def _compute_combined_reward_for_group(
    metrics_list: list,
    reward_weights: list = None,
) -> list:
    """
    Compute the combined reward score for a group of sequences from the same
    structure, using the SAME normalisation logic as training reward functions.

    Args:
        metrics_list: list of tuples — supported shapes:
                      6-element
                        (seq, rr, tm, scrmsd, plddt, ddg_raw)
                        [validation]
                      7-element
                        (seq, rr, tm, scrmsd, plddt, ddg_raw, reward)
                        [training]
        reward_weights: list of (name, weight) pairs, e.g.
                        [("tm", 1.0), ("ddg", 1.0), ("plddt", 0.0),
                         ("recovery", 0.0), ("length", 0.0)]
                        If None, falls back to sorting by pLDDT.

    Returns:
        list of combined reward scores (one per entry in metrics_list),
        in the same order as the input.

    Degenerate-group convention: _minmax_norm returns 0.5 when hi == lo,
    while the training reward divides by (hi - lo + 1e-8) and therefore
    returns 0.0. The main weighted-sum formula is identical; GRPO group
    centering makes the constant degenerate-group offset immaterial.
    """
    n = len(metrics_list)
    if n == 0:
        return []

    # Use pLDDT when no validation reward weights are configured.
    if not reward_weights or all(w == 0.0 for _, w in reward_weights):
        return [m[4] for m in metrics_list]

    # Extract raw values per reward component
    tm_vals   = [m[2] for m in metrics_list]
    ddg_vals  = [m[5] for m in metrics_list]
    plddt_vals = [m[4] for m in metrics_list]
    rr_vals   = [m[1] for m in metrics_list]

    # Per-group min/max for components that need group normalization
    tm_lo, tm_hi     = min(tm_vals), max(tm_vals)
    ddg_lo, ddg_hi   = min(ddg_vals), max(ddg_vals)

    combined_scores = []
    for i in range(n):
        score = 0.0
        for name, weight in reward_weights:
            if weight == 0.0:
                continue
            if name == "tm":
                # Higher raw values receive higher min-max-normalized scores.
                score += weight * _minmax_norm(tm_vals[i], tm_lo, tm_hi)
            elif name == "ddg":
                # Lower raw values receive higher min-max-normalized scores.
                score += weight * (1.0 - _minmax_norm(ddg_vals[i], ddg_lo, ddg_hi))
            elif name == "plddt":
                # Scale pLDDT to [0, 1].
                score += weight * (plddt_vals[i] / 100.0)
            elif name == "recovery":
                # The recovery rate is already bounded to [0, 1].
                score += weight * rr_vals[i]
            elif name == "length":
                # Exact-length validation assigns a constant length contribution.
                score += weight * 1.0
        combined_scores.append(score)

    return combined_scores


# Fixed weighted multi-objective reward

def multi_objective_reward(completions: list, structure_id: list = None, **kwargs) -> list:
    """
    Paper Sec. 3.1.2 multi-objective reward.

    Raw objectives are min-max normalized within each backbone candidate pool,
    with higher values preferred for ordinary objectives and lower physical
    ΔΔG values preferred for the stability objective. The normalized values
    are then combined by a fixed weighted sum. The weights are not normalized
    by their sum. With scale_rewards="group", GRPO z-scoring absorbs a
    positive common scale; with scale_rewards="none" (Dr.GRPO), their
    absolute scale intentionally changes the effective update scale.
    """
    native_seq_list = kwargs.get("native_seq", [None] * len(completions))
    native_seq_len_list = kwargs.get("native_seq_len", [None] * len(completions))

    user_w = {
        name: weight
        for name, weight in (REWARD_WEIGHTS_GLOBAL or [])
        if weight != 0.0
    }
    enabled = list(user_w)
    if not enabled:
        try:
            _accumulate_train_metrics(completions, structure_id, native_seq_list)
        except Exception as e:
            logger.warning(f"_accumulate_train_metrics failed: {e}")
        return [0.0] * len(completions)

    n = len(completions)
    raw = {name: [] for name in enabled}

    for i in range(n):
        comp = completions[i]
        sid = structure_id[i] if structure_id else None

        seq = clean(comp)
        if not seq or len(seq) < 10:
            for name in enabled:
                raw[name].append(0.0)
            continue

        for name in enabled:
            try:
                if name == "tm":
                    raw[name].append(float(fast_tm_score(seq, sid)))
                elif name == "ddg":
                    raw[name].append(float(fast_ddg(seq, sid)))
                elif name == "plddt":
                    raw[name].append(float(fast_plddt(seq)) / 100.0)
                elif name == "recovery":
                    nat = native_seq_list[i]
                    raw[name].append(float(partial_recovery(seq, nat)))
                elif name == "length":
                    target_len = native_seq_len_list[i]
                    if target_len is None:
                        raw[name].append(0.0)
                    else:
                        actual_len = len(seq)
                        if actual_len == target_len:
                            raw[name].append(1.0)
                        else:
                            mismatch_ratio = abs(actual_len - target_len) / max(target_len, 1)
                            raw[name].append(float(math.exp(-5.0 * mismatch_ratio) - 1.0))
                else:
                    raw[name].append(0.0)
            except Exception as e:
                logger.warning(f"reward: {name} failed for {sid}: {e}")
                raw[name].append(0.0)

    # Gather raw rewards across ranks for consistent normalization.
    local_n = n
    if dist.is_initialized() and dist.get_world_size() > 1:
        all_raw = [None] * dist.get_world_size()
        all_sids = [None] * dist.get_world_size()
        dist.all_gather_object(all_raw, raw)
        dist.all_gather_object(all_sids, list(structure_id))

        rank = dist.get_rank()
        g_raw = {name: [] for name in enabled}
        g_sids = []
        rank_offsets = []
        for r in range(dist.get_world_size()):
            rank_offsets.append(len(g_sids))
            g_sids.extend(all_sids[r])
            for name in enabled:
                g_raw[name].extend(all_raw[r][name])
    else:
        g_raw = raw
        g_sids = list(structure_id)
        rank_offsets = [0]
        rank = 0

    from collections import defaultdict
    groups = defaultdict(list)
    for idx, sid in enumerate(g_sids):
        groups[sid].append(idx)

    total = len(g_sids)
    rewards = [0.0] * total
    delta = 1e-8

    for sid, indices in groups.items():
        # Per-group min-max normalization. Physical ΔΔG is lower-is-better;
        # every other enabled objective is higher-is-better.
        normed = {}
        raw_ranges = {}
        for name in enabled:
            vals = [g_raw[name][i] for i in indices]
            lo, hi = min(vals), max(vals)
            raw_ranges[name] = (lo, hi)
            normed[name] = [
                _training_objective_norm(
                    name, g_raw[name][i], lo, hi, delta
                )
                for i in indices
            ]

        # Fixed linear combination of direction-aware, per-backbone normalized objectives.
        group_rewards = []
        for j, i in enumerate(indices):
            rewards[i] = sum(
                user_w[name] * normed[name][j] for name in enabled
            )
            group_rewards.append(rewards[i])

        # Per-group diagnostics.
        w_str = " ".join(f"{name}={user_w[name]:.3f}" for name in enabled)
        range_str = " ".join(
            f"{name}=[{raw_ranges[name][0]:.3f},{raw_ranges[name][1]:.3f}]"
            for name in enabled
        )
        avg_reward = sum(group_rewards) / len(group_rewards)
        logger.info(
            f"  [reward] {sid} ({len(indices)} seqs): "
            f"weights {{ {w_str} }}  avg_reward={avg_reward:.4f}  "
            f"raw_ranges {{ {range_str} }}"
        )

    my_start = rank_offsets[rank]
    local_rewards = rewards[my_start:my_start + local_n]

    # Log per-sequence metrics with the final linear reward.
    # Pass local raw ΔΔG values to avoid redundant scorer evaluation.
    local_reward_values = list(local_rewards)
    local_ddg = raw.get("ddg", None)
    try:
        _accumulate_train_metrics(
            completions,
            structure_id,
            native_seq_list,
            reward_values=local_reward_values,
            ddg_values=local_ddg,
        )
    except Exception as e:
        logger.warning(f"_accumulate_train_metrics failed: {e}")

    return local_rewards


# Exact-length batched generation

class ForceExactLengthBatch(LogitsProcessor):
    """
    Forces each sequence in a batch to have EXACTLY the target number of
    amino-acid residues, then emits the sentinel token "2".

    Per-row target lengths support heterogeneous batches.

    Behaviour:
      - Before target_len residues: block sentinel "2" and all non-AA tokens.
      - At exactly target_len residues: force sentinel "2".
      - After that: let EOS handling take over.

    prefix_len is auto-detected on the first call (= input_ids.size(1)
    at step 0, before any new token is appended).
    """

    def __init__(self, action_space, target_lens):
        self.action_space = action_space
        self.sentinel_id = action_space.sentinel2_id
        self.target_lens = [int(length) for length in target_lens]
        self.aa_token_ids = action_space.aa_token_id_set
        self._prefix_len = None
        if not self.target_lens or any(length <= 0 for length in self.target_lens):
            raise ValueError(f"Invalid target lengths: {self.target_lens}")

    def __call__(self, input_ids, scores):
        if self._prefix_len is None:
            self._prefix_len = input_ids.size(1)

        cur_len = input_ids.size(1)
        n_new = cur_len - self._prefix_len          # tokens generated so far

        if input_ids.size(0) != len(self.target_lens):
            raise RuntimeError(
                f"Generation rows {input_ids.size(0)} != targets {len(self.target_lens)}"
            )
        for i, target in enumerate(self.target_lens):

            if n_new < target:
                scores[i, self.sentinel_id] = -float("inf")
                if self.aa_token_ids:
                    mask = torch.full_like(scores[i], -float("inf"))
                    for aa_id in self.aa_token_ids:
                        mask[aa_id] = 0.0
                    scores[i] = scores[i] + mask

            else:
                scores[i, :] = -float("inf")
                scores[i, self.sentinel_id] = 0.0

        return scores


# GRPO trainer with exact-length generation and diversity regularization

class DiversityGroupingError(RuntimeError):
    """Grouping invariant violation; must never be silently swallowed."""


class ProteinGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer extended with:
      1. Exact-length logits processing during generation
      2. Embedding-space diversity penalty in the loss

    ForceExactLength ensures that GRPO-generated sequences have EXACTLY
    the same length as the native backbone sequence. This is critical for
    backbone-conditioned protein design — without it, recovery rate is
    near-zero because position-wise comparison fails on length-mismatched
    sequences.
    """

    def __init__(self, *args, alpha_diversity=0.05, force_exact_length=True,
                 sid_to_target_info=None, protein_action_space=None,
                 policy_provenance=None, **kwargs):
        if not force_exact_length or not sid_to_target_info or protein_action_space is None:
            raise RuntimeError(
                "Protein GRPO requires exact length, target lengths, and ProteinActionSpace"
            )
        self.force_exact_length = True
        if not isinstance(policy_provenance, dict) or "mode" not in policy_provenance:
            raise RuntimeError("ProteinGRPOTrainer requires explicit policy provenance")
        self.sid_to_target_info = sid_to_target_info
        ordered_sids = sorted(self.sid_to_target_info)
        self._sid_to_structure_index = {
            sid: index for index, sid in enumerate(ordered_sids)
        }
        self._structure_index_to_sid = {
            index: sid for sid, index in self._sid_to_structure_index.items()
        }
        self.protein_action_space = protein_action_space
        self.policy_provenance = copy.deepcopy(policy_provenance)
        super().__init__(*args, **kwargs)
        self.alpha_diversity = alpha_diversity
        self._is_new_backbone = False
        self._strict_runtime_checked = False
        self._setup_exact_length()
        logger.info(f"[ProteinGRPOTrainer] exact length for {len(sid_to_target_info)} SIDs")

        # Pairwise diversity: accumulate one complete generation batch across
        # micro-batches, then group its rows by SID and re-forward each sequence
        # to compute within-SID cosine diversity with full gradient flow.
        self._div_input_buffer = []
        self._is_last_micro_of_backbone = False
        if alpha_diversity > 0:
            logger.info(f"[ProteinGRPOTrainer] Pairwise diversity penalty enabled: α={alpha_diversity}")
        if self.is_world_process_zero():
            logger.info(
                f"[diversity] steps_per_generation={self.args.steps_per_generation} "
                f"generation_batch_size={self.args.generation_batch_size} "
                f"num_generations={self.num_generations} "
                f"expected_groups="
                f"{self.args.generation_batch_size // self.num_generations} "
                f"per_device_bs={self.args.per_device_train_batch_size} "
                f"world_size={self.args.world_size}"
            )
        self._enforce_and_audit_policy_runtime("trainer_init")

    def _enforce_and_audit_policy_runtime(self, stage, require_no_amp_wrapper=False):
        false_fields = ("fp16", "bf16", "fp16_full_eval", "bf16_full_eval", "tf32")
        missing = object()
        failures = []
        for name in false_fields:
            actual = getattr(self.args, name, missing)
            if actual is missing or bool(actual):
                actual_repr = "<MISSING>" if actual is missing else repr(actual)
                actual_type = "MISSING" if actual is missing else type(actual).__name__
                failures.append(
                    f"{name}: expected=False, actual={actual_repr}, type={actual_type}"
                )
        disable_dropout = getattr(self.args, "disable_dropout", missing)
        if disable_dropout is missing or not bool(disable_dropout):
            actual_repr = "<MISSING>" if disable_dropout is missing else repr(disable_dropout)
            actual_type = "MISSING" if disable_dropout is missing else type(disable_dropout).__name__
            failures.append(
                "disable_dropout: expected=True, "
                f"actual={actual_repr}, type={actual_type}"
            )
        mixed_precision = getattr(self.args, "mixed_precision", missing)
        if mixed_precision != "no":
            actual_repr = "<MISSING>" if mixed_precision is missing else repr(mixed_precision)
            actual_type = "MISSING" if mixed_precision is missing else type(mixed_precision).__name__
            failures.append(
                "mixed_precision: expected='no', "
                f"actual={actual_repr}, type={actual_type}"
            )
        if failures:
            raise RuntimeError(
                f"[{stage}] invalid precision/dropout args: " + "; ".join(failures)
            )
        if self.is_deepspeed_enabled or self.is_fsdp_enabled:
            raise RuntimeError("Strict LM FP32 supports only single-process/DDP execution")
        n_gpu = getattr(self.args, "n_gpu", 1)
        if n_gpu > 1:
            raise RuntimeError(
                f"[{stage}] n_gpu={n_gpu} triggers nn.DataParallel, which breaks the "
                "strict FP32 / dropout / adapter guarantees. Use torchrun for multi-GPU, "
                "or set CUDA_VISIBLE_DEVICES to a single device."
            )
        if isinstance(getattr(self, "model_wrapped", None), torch.nn.DataParallel):
            raise RuntimeError(f"[{stage}] model is wrapped in nn.DataParallel")
        if self.rollout_func is not None or self.tools:
            raise RuntimeError("Custom rollout functions and tools are unsupported")
        if (self.accelerator.mixed_precision != "no" or self.accelerator.native_amp
                or self.accelerator.scaler is not None):
            raise RuntimeError(f"[{stage}] unexpected Accelerate AMP state")
        assert_ieee_fp32_runtime()

        wrapped = getattr(self, "model_wrapped", self.model)
        policy = self.accelerator.unwrap_model(wrapped)
        models = [("policy", policy)]
        if self.ref_model is not None:
            models.append(("reference", self.accelerator.unwrap_model(self.ref_model)))
        for role, model in models:
            if require_no_amp_wrapper and "_original_forward" in model.__dict__:
                raise RuntimeError(
                    f"Accelerate installed a mixed-precision wrapper on {role}"
                )
            dropouts = [(name, module) for name, module in model.named_modules()
                        if isinstance(module, torch.nn.Dropout)]
            nonzero_before = sum(module.p != 0 for _, module in dropouts)
            for _, module in dropouts:
                module.p = 0.0
            nonzero_after = sum(module.p != 0 for _, module in dropouts)
            if nonzero_after:
                raise RuntimeError(f"[{stage}] failed to disable every runtime dropout")
            lora_dropouts = [(name, module) for name, module in dropouts
                             if "lora_dropout" in name]
            runtime = {}
            for name, module in lora_dropouts:
                runtime.setdefault(name.rsplit(".", 1)[-1], []).append(float(module.p))
            runtime = {name: dict(Counter(values)) for name, values in runtime.items()}
            configs = {name: getattr(value, "lora_dropout", None)
                       for name, value in getattr(model, "peft_config", {}).items()}
            dtypes = Counter(str(p.dtype) for p in model.parameters() if p.is_floating_point())
            buffer_dtypes = Counter(
                str(buffer.dtype) for buffer in model.buffers()
                if buffer.is_floating_point()
            )
            if set(dtypes) != {"torch.float32"}:
                raise RuntimeError(f"[{stage}] {role} parameter dtypes={dict(dtypes)}")
            if set(buffer_dtypes) - {"torch.float32"}:
                raise RuntimeError(
                    f"[{stage}] {role} floating buffer dtypes={dict(buffer_dtypes)}"
                )
            logger.info(
                f"[LM runtime:{stage}:{role}] dropout_total={len(dropouts)} "
                f"lora_dropout={len(lora_dropouts)} nonzero_before={nonzero_before} "
                f"nonzero_after={nonzero_after} "
                f"active={list(getattr(model, 'active_adapters', []))} "
                f"config_p={configs} runtime_p={runtime} dtypes={dict(dtypes)} "
                f"buffer_dtypes={dict(buffer_dtypes)}"
            )
        logger.info(
            f"[LM runtime:{stage}] mixed_precision={self.accelerator.mixed_precision} "
            f"native_amp={self.accelerator.native_amp} scaler={self.accelerator.scaler} "
            f"fp32_precision={getattr(torch.backends, 'fp32_precision', 'legacy-ieee')}"
        )

    # ForceExactLength helpers

    def _setup_exact_length(self):
        """Install the immutable protein EOS/action-space contract."""
        spec = self.protein_action_space
        self._aa_token_ids = spec.aa_token_id_set
        self._sentinel2_id = spec.sentinel2_id
        self.eos_token_id = spec.sentinel2_id
        model_eos = getattr(self.model.config, "eos_token_id", None)
        if model_eos != spec.special_eos_id:
            raise RuntimeError(
                f"Model special EOS {model_eos} != action-space EOS {spec.special_eos_id}"
            )
        if self.generation_config is None:
            raise RuntimeError("Missing generation_config")
        gc = self.generation_config
        model_gc = getattr(self.model, "generation_config", None)
        # Transformers 5.1 generation/configuration_utils.py:538-555 supplies
        # these defaults only inside generate(). Resolve the same precedence
        # here: trainer GenerationConfig -> model GenerationConfig -> HF default.
        generation_defaults = {
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "use_cache": True,
        }
        expected = {
            "do_sample": True,
            "num_beams": 1,
            "num_return_sequences": 1,
            "use_cache": True,
        }
        effective = {}
        observed = {}
        sources = {}
        for name, default in generation_defaults.items():
            trainer_value = getattr(gc, name, None)
            model_value = getattr(model_gc, name, None) if model_gc is not None else None
            if trainer_value is not None:
                effective[name] = trainer_value
                sources[name] = "trainer_generation_config"
            elif model_value is not None:
                effective[name] = model_value
                sources[name] = "model_generation_config"
            else:
                effective[name] = default
                sources[name] = "transformers_default"
            observed[name] = trainer_value

        generation_checks = {
            "do_sample": bool(effective["do_sample"]),
            "num_beams": effective["num_beams"] == 1,
            "num_return_sequences": effective["num_return_sequences"] == 1,
            "use_cache": bool(effective["use_cache"]),
        }
        failures = []
        for name, passed in generation_checks.items():
            if not passed:
                raw = observed[name]
                value = effective[name]
                failures.append(
                    f"{name}: expected={expected[name]!r}, raw={raw!r}, "
                    f"raw_type={type(raw).__name__}, effective={value!r}, "
                    f"effective_type={type(value).__name__}, source={sources[name]}"
                )
        if failures:
            raise RuntimeError(
                "Unsupported generation contract: " + "; ".join(failures)
            )
        self.generation_config.eos_token_id = [spec.special_eos_id, spec.sentinel2_id]
        logger.info(
            f"Protein generation EOS={self.generation_config.eos_token_id}; "
            f"A20={spec.aa_token_ids}"
        )

    def _target_lengths(self, prompts):
        if not prompts:
            raise RuntimeError("Protein generation received an empty prompt batch")
        target_lens = []
        for prompt in prompts:
            sid = parse_structure_prompt(prompt)
            info = self.sid_to_target_info.get(sid)
            if info is None:
                raise RuntimeError(f"Unknown structure_id {sid!r} with exact length enabled")
            target = int(info["target_len"])
            if target <= 0:
                raise RuntimeError(f"Invalid target length for {sid}: {target}")
            if target + 1 > self.max_completion_length:
                raise RuntimeError(
                    f"Target {sid} needs {target + 1} completion tokens, "
                    f"limit is {self.max_completion_length}"
                )
            target_lens.append(target)
        return target_lens

    def _build_logits_processor(self, prompts):
        target_lens = self._target_lengths(prompts)
        return LogitsProcessorList([
            ForceExactLengthBatch(self.protein_action_space, target_lens)
        ])

    # Exact rollout and constrained replay

    def _generate_single_turn(self, prompts):
        """Generate exact-length proteins and retain true processed-score log-probs."""
        target_lens = self._target_lengths(prompts)
        logits_processor = self._build_logits_processor(prompts)
        device = self.accelerator.device
        assert_ieee_fp32_runtime()
        if torch.is_autocast_enabled(device.type):
            raise RuntimeError("Rollout entered with autocast enabled")

        generate_inputs = self.processing_class(
            text=prompts, padding=True, padding_side="left", return_tensors="pt"
        )
        prompt_shape = tuple(generate_inputs["input_ids"].shape)
        if prompt_shape != (len(prompts), 257):
            raise RuntimeError(f"Protein prompt tensor must be [B,257], got {prompt_shape}")
        if not generate_inputs["input_ids"][:, -1].eq(
            self.protein_action_space.sentinel1_id
        ).all():
            raise RuntimeError("Protein prompt must end in sentinel1 token ID 3")
        generate_inputs = super(GRPOTrainer, self)._prepare_inputs(generate_inputs)
        if "attention_mask" not in generate_inputs:
            raise RuntimeError("Tokenizer did not return attention_mask")
        generate_inputs["attention_mask"] = torch.ones_like(generate_inputs["attention_mask"])

        with (
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model,
            torch.no_grad(),
            torch.autocast(device_type=device.type, enabled=False),
        ):
            generation_output = unwrapped_model.generate(
                **generate_inputs,
                generation_config=self.generation_config,
                logits_processor=logits_processor,
                return_dict_in_generate=True,
                output_scores=True,
                disable_compile=True,
            )

        prompt_ids_tensor = generate_inputs["input_ids"]
        prompt_mask = generate_inputs["attention_mask"]
        prompt_length = prompt_ids_tensor.size(1)
        completion_ids_tensor = generation_output.sequences[:, prompt_length:]
        if not generation_output.scores:
            raise RuntimeError("generate() returned no processed sampling scores")
        score_dtypes = {score.dtype for score in generation_output.scores}
        if score_dtypes != {torch.float32}:
            raise RuntimeError(f"Rollout processed scores must be FP32, got {score_dtypes}")
        processed_scores = torch.stack(tuple(generation_output.scores), dim=1)
        if processed_scores.shape[:2] != completion_ids_tensor.shape:
            raise RuntimeError(
                f"Sampling score shape {processed_scores.shape} != IDs {completion_ids_tensor.shape}"
            )
        behavior_logps = F.log_softmax(processed_scores, dim=-1).gather(
            -1, completion_ids_tensor.unsqueeze(-1)
        ).squeeze(-1)

        is_sentinel = completion_ids_tensor.eq(self._sentinel2_id)
        sentinel_count = is_sentinel.sum(dim=1)
        if not torch.equal(sentinel_count, torch.ones_like(sentinel_count)):
            raise RuntimeError(f"Expected one forced sentinel per row, got {sentinel_count.tolist()}")
        eos_idx = is_sentinel.int().argmax(dim=1)
        expected_idx = torch.tensor(target_lens, device=device)
        if not torch.equal(eos_idx, expected_idx):
            raise RuntimeError(
                f"Sentinel indices {eos_idx.tolist()} != targets {target_lens}"
            )
        sequence_indices = torch.arange(
            completion_ids_tensor.size(1), device=device
        ).expand_as(completion_ids_tensor)
        sequence_mask = sequence_indices <= eos_idx.unsqueeze(1)
        retained_logps = behavior_logps[sequence_mask]
        if not torch.isfinite(retained_logps).all():
            raise RuntimeError("Non-finite behavior log-prob on a retained token")
        sentinel_logps = behavior_logps.gather(1, eos_idx.unsqueeze(1)).squeeze(1)
        if not torch.equal(sentinel_logps, torch.zeros_like(sentinel_logps)):
            raise RuntimeError(f"Forced sentinel log-prob must be exactly zero: {sentinel_logps}")
        allowed = torch.tensor(self.protein_action_space.aa_token_ids, device=device)
        aa_prefix = sequence_indices < eos_idx.unsqueeze(1)
        if not torch.isin(completion_ids_tensor[aa_prefix], allowed).all():
            raise RuntimeError("Rollout retained a non-A20 residue action")

        prompt_ids = [
            row[mask.bool()].tolist()
            for row, mask in zip(prompt_ids_tensor, prompt_mask, strict=True)
        ]
        completion_ids = [
            row[mask].tolist()
            for row, mask in zip(completion_ids_tensor, sequence_mask, strict=True)
        ]
        sampling_logps = [
            row[mask].tolist()
            for row, mask in zip(behavior_logps, sequence_mask, strict=True)
        ]
        return prompt_ids, completion_ids, sampling_logps, {}

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ):
        """TRL 0.27.2 hook: strict-FP32, A20-normalized replay/reference."""
        assert_ieee_fp32_runtime()
        if torch.is_autocast_enabled(input_ids.device.type):
            raise RuntimeError("Replay entered with autocast enabled")
        if any(value is not None for value in (
            pixel_values, image_grid_thw, num_images, pixel_attention_mask,
            image_sizes, token_type_ids,
        )):
            raise RuntimeError("ProteinGRPOTrainer does not support multimodal model inputs")
        batch_size = batch_size or input_ids.size(0)
        all_logps, all_entropies = [], []
        for start in range(0, input_ids.size(0), batch_size):
            ids = input_ids[start:start + batch_size]
            mask = attention_mask[start:start + batch_size].clone()
            completion_ids = ids[:, -logits_to_keep:]
            prompt_width = ids.size(1) - logits_to_keep
            # TRL 0.27.2 trl/trainer/grpo_trainer.py:1849-1853 reconstructs prompt_mask
            # from returned list lengths. The 256 structure slots plus sentinel1
            # must therefore arrive here as 257 visible prompt positions.
            if prompt_width != 257:
                raise RuntimeError(f"Protein replay prompt width must be 257, got {prompt_width}")
            if not torch.equal(
                mask[:, :prompt_width], torch.ones_like(mask[:, :prompt_width])
            ):
                raise RuntimeError("Protein replay prompt mask must be all ones")
            # action_mask excludes the forced sentinel; the model must still see it.
            mask[:, -logits_to_keep:] |= completion_ids.eq(
                self.protein_action_space.sentinel2_id
            ).to(mask.dtype)
            model_inputs = {"input_ids": ids, "attention_mask": mask, "use_cache": False}
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            with torch.autocast(device_type=ids.device.type, enabled=False):
                logits = model(**model_inputs).logits
            if logits.dtype != torch.float32:
                raise RuntimeError(f"Replay logits must be FP32, got {logits.dtype}")
            logits = logits[:, :-1, :][:, -logits_to_keep:, :]
            logps, entropies = protein_action_logps_and_entropies(
                logits,
                completion_ids,
                self.protein_action_space,
                self.temperature,
                compute_entropy=compute_entropy,
            )
            all_logps.append(logps)
            if compute_entropy:
                all_entropies.append(entropies)
        return (
            torch.cat(all_logps, dim=0),
            torch.cat(all_entropies, dim=0) if compute_entropy else None,
        )

    def _generate_and_score_completions(self, inputs):
        """Attach the residue action mask and make captured behavior log-probs old."""
        output = super()._generate_and_score_completions(inputs)
        prompt_mask = output["prompt_mask"]
        if prompt_mask.size(1) != 257 or not torch.equal(
            prompt_mask, torch.ones_like(prompt_mask)
        ):
            raise RuntimeError("TRL reconstructed a non-all-one 257-position prompt mask")
        completion_ids = output["completion_ids"]
        sequence_mask = output["completion_mask"].bool()
        # TRL 0.27.2 trl/trainer/grpo_trainer.py:1845,1858-1862,2133-2134 carries
        # _generate_single_turn's third return into this output when non-None.
        sampling_logps = output.get("sampling_per_token_logps")
        if sampling_logps is None or sampling_logps.shape != completion_ids.shape:
            raise RuntimeError("Missing or malformed rollout behavior log-probs")
        if len(inputs) != completion_ids.size(0):
            raise RuntimeError(
                f"Input rows {len(inputs)} != completion rows {completion_ids.size(0)}"
            )

        # TRL 0.27.2 trl/trainer/grpo_trainer.py:2121-2151 returns a
        # whitelist that drops dataset-only columns. Reattach a stable numeric
        # SID before utils.py:992-1021 and :957-989 jointly shuffle and split
        # every rank-1+ tensor along its leading batch dimension.
        structure_indices = []
        for row_number, input_row in enumerate(inputs):
            prompt_sid = parse_structure_prompt(input_row.get("prompt"))
            supplied_sid = input_row.get("structure_id")
            if supplied_sid is not None and supplied_sid != prompt_sid:
                raise DiversityGroupingError(
                    f"Row {row_number} structure_id {supplied_sid!r} "
                    f"!= prompt SID {prompt_sid!r}"
                )
            structure_index = self._sid_to_structure_index.get(prompt_sid)
            if structure_index is None:
                raise DiversityGroupingError(
                    f"Row {row_number} has unmapped structure_id {prompt_sid!r}"
                )
            structure_indices.append(structure_index)
        structure_index = torch.tensor(
            structure_indices, dtype=torch.long, device=completion_ids.device
        )
        if structure_index.shape != (completion_ids.size(0),):
            raise DiversityGroupingError(
                f"structure_index shape {tuple(structure_index.shape)} "
                f"!= completion batch {(completion_ids.size(0),)}"
            )

        target_lens = self._target_lengths([row["prompt"] for row in inputs])
        action_mask = torch.zeros_like(output["completion_mask"])
        aa_ids = torch.tensor(self.protein_action_space.aa_token_ids, device=completion_ids.device)
        for row, target in enumerate(target_lens):
            retained_positions = sequence_mask[row].nonzero(as_tuple=True)[0]
            retained = completion_ids[row][retained_positions]
            if retained.numel() != target + 1:
                raise RuntimeError(
                    f"Row {row}: retained {retained.numel()} tokens, expected {target + 1}"
                )
            if retained[-1].item() != self.protein_action_space.sentinel2_id:
                raise RuntimeError(f"Row {row}: forced sentinel is not the final retained token")
            if not torch.isin(retained[:-1], aa_ids).all():
                raise RuntimeError(f"Row {row}: non-A20 residue in rollout")
            if (retained == self.protein_action_space.sentinel2_id).sum().item() != 1:
                raise RuntimeError(f"Row {row}: sentinel count is not one")
            action_mask[row, retained_positions[:-1]] = 1
            sentinel_position = retained_positions[-1]
            if sampling_logps[row, sentinel_position].item() != 0.0:
                raise RuntimeError(f"Row {row}: forced sentinel behavior log-prob is not zero")

        if not torch.equal(action_mask.sum(dim=1).cpu(), torch.tensor(target_lens)):
            raise RuntimeError("Policy action count does not match target residue length")
        if not torch.isfinite(sampling_logps[action_mask.bool()]).all():
            raise RuntimeError("Non-finite behavior log-prob on policy action")
        output["action_mask"] = action_mask
        output["structure_index"] = structure_index
        # TRL 0.27.2 trl/trainer/grpo_trainer.py:1906-1928 may skip its
        # old-policy forward; :2276-2277 nevertheless preserves this non-None
        # tensor and falls back to current.detach() only for missing/None.
        output["old_per_token_logps"] = sampling_logps.masked_fill(
            ~action_mask.bool(), 0.0
        ).detach().clone()
        return output

    # Pairwise diversity penalty (full gradient, cross-GPU)

    def _compute_loss(self, model, inputs):
        if "action_mask" not in inputs:
            raise RuntimeError("Missing protein action_mask")
        rl_inputs = dict(inputs)
        rl_inputs["completion_mask"] = inputs["action_mask"]
        # TRL 0.27.2 trl/trainer/grpo_trainer.py:2238-2250 consumes this
        # completion mask for current replay and all downstream token averages.
        loss = super()._compute_loss(model, rl_inputs)

        # Evaluation does not use the training generation-round buffer; evaluation
        # rows therefore bypass buffer mutation and flushing.
        if not model.training:
            return loss
        if self.alpha_diversity < 1e-8:
            return loss

        try:
            # Phase 1: accumulate one generation round
            if self._is_new_backbone and self._div_input_buffer:
                raise DiversityGroupingError(
                    "New generation round started with a non-empty diversity "
                    f"buffer ({len(self._div_input_buffer)} micro-batches)"
                )

            input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
            attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
            structure_index = inputs.get("structure_index")
            if not isinstance(structure_index, torch.Tensor):
                raise DiversityGroupingError(
                    "Missing tensor structure_index in diversity inputs"
                )
            if structure_index.dtype != torch.long or structure_index.ndim != 1:
                raise DiversityGroupingError(
                    "structure_index must be rank-1 torch.long, got "
                    f"dtype={structure_index.dtype}, "
                    f"shape={tuple(structure_index.shape)}"
                )
            if structure_index.numel() != input_ids.size(0):
                raise DiversityGroupingError(
                    f"structure_index rows {structure_index.numel()} "
                    f"!= input rows {input_ids.size(0)}"
                )
            self._div_input_buffer.append((
                input_ids.detach().cpu(),
                attention_mask.detach().cpu(),
                structure_index.detach().cpu(),
            ))

            # Phase 2: flush only at the generation-round boundary
            if not self._is_last_micro_of_backbone:
                return loss

            spg = int(self.args.steps_per_generation)
            local_inputs = tuple(self._div_input_buffer)
            self._div_input_buffer.clear()

            # Validate local tuple and row shapes before the collective. Propagate local
            # failures through the collective so that every rank raises the same error.
            rank = dist.get_rank() if dist.is_initialized() else 0
            local_errors = []
            if len(local_inputs) != spg:
                local_errors.append(
                    f"local buffer has {len(local_inputs)} micro-batches, "
                    f"expected steps_per_generation={spg}"
                )
            if self._div_input_buffer:
                local_errors.append(
                    "diversity buffer was not empty immediately after flush"
                )
            for chunk_number, packed in enumerate(local_inputs):
                if not isinstance(packed, (list, tuple)) or len(packed) != 3:
                    local_errors.append(
                        f"chunk {chunk_number} is not a three-tensor tuple"
                    )
                    continue
                ids_cpu, mask_cpu, index_cpu = packed
                if not isinstance(ids_cpu, torch.Tensor) or ids_cpu.ndim != 2:
                    local_errors.append(
                        f"chunk {chunk_number} has invalid input_ids"
                    )
                    continue
                if (
                    not isinstance(mask_cpu, torch.Tensor)
                    or mask_cpu.shape != ids_cpu.shape
                ):
                    local_errors.append(
                        f"chunk {chunk_number} attention mask shape mismatch"
                    )
                if (
                    not isinstance(index_cpu, torch.Tensor)
                    or index_cpu.dtype != torch.long
                    or index_cpu.ndim != 1
                    or index_cpu.numel() != ids_cpu.size(0)
                ):
                    local_errors.append(
                        f"chunk {chunk_number} has malformed structure_index"
                    )

            local_payload = {
                "rank": rank,
                "errors": local_errors,
                "inputs": local_inputs,
            }
            if dist.is_initialized() and dist.get_world_size() > 1:
                rank_payloads = [None] * dist.get_world_size()
                dist.all_gather_object(rank_payloads, local_payload)
            else:
                rank_payloads = [local_payload]

            gathered_errors = []
            all_inputs = []
            for payload in rank_payloads:
                if not isinstance(payload, dict):
                    raise DiversityGroupingError(
                        f"Invalid gathered payload type {type(payload).__name__}"
                    )
                payload_rank = payload.get("rank", "<missing>")
                gathered_errors.extend(
                    f"rank {payload_rank}: {message}"
                    for message in payload.get("errors", [])
                )
                payload_inputs = payload.get("inputs")
                if not isinstance(payload_inputs, (list, tuple)):
                    raise DiversityGroupingError(
                        f"Rank {payload_rank} returned invalid diversity inputs"
                    )
                all_inputs.extend(payload_inputs)
            if gathered_errors:
                raise DiversityGroupingError(
                    "Diversity local grouping invariant failed: "
                    + "; ".join(gathered_errors)
                )

            # len(all_inputs) counts micro-batch tuples, not necessarily rows.
            index_parts = []
            global_n_seqs = 0
            for chunk_number, (ids_cpu, mask_cpu, index_cpu) in enumerate(all_inputs):
                # All gathered payloads have passed the same local shape validation.
                global_n_seqs += ids_cpu.size(0)
                index_parts.append(index_cpu)

            generation_batch_size = int(self.args.generation_batch_size)
            if global_n_seqs != generation_batch_size:
                raise DiversityGroupingError(
                    f"Gathered {global_n_seqs} sequences, expected "
                    f"generation_batch_size={generation_batch_size}"
                )

            num_generations = int(self.num_generations)
            if num_generations < 2:
                raise DiversityGroupingError(
                    f"Diversity requires num_generations>=2, got {num_generations}"
                )
            if generation_batch_size % num_generations != 0:
                raise DiversityGroupingError(
                    f"generation_batch_size={generation_batch_size} is not "
                    f"divisible by num_generations={num_generations}"
                )
            expected_num_groups = generation_batch_size // num_generations
            all_structure_indices_cpu = torch.cat(index_parts, dim=0)
            unique_indices_cpu, group_counts_cpu = torch.unique(
                all_structure_indices_cpu, sorted=True, return_counts=True
            )
            observed_counts = {
                self._structure_index_to_sid.get(
                    int(index), f"<unknown-index:{int(index)}>"
                ): int(count)
                for index, count in zip(
                    unique_indices_cpu.tolist(),
                    group_counts_cpu.tolist(),
                    strict=True,
                )
            }
            if unique_indices_cpu.numel() != expected_num_groups:
                raise DiversityGroupingError(
                    f"Gathered {unique_indices_cpu.numel()} SID groups, expected "
                    f"{expected_num_groups}; counts={observed_counts}"
                )
            bad_counts = {
                sid: count
                for sid, count in observed_counts.items()
                if count != num_generations
            }
            if bad_counts:
                raise DiversityGroupingError(
                    f"Each SID group must contain num_generations="
                    f"{num_generations}; bad counts={bad_counts}"
                )

            device = input_ids.device
            all_embs = []
            for ids_cpu, mask_cpu, _ in all_inputs:
                ids = ids_cpu.to(device)
                mask = mask_cpu.to(device)
                out = model(input_ids=ids, attention_mask=mask,
                            output_hidden_states=True, use_cache=False)
                hs = out.hidden_states[-1]
                mask_3d = mask.unsqueeze(-1).float()
                emb = (hs * mask_3d).sum(1) / mask_3d.sum(1).clamp(min=1)
                all_embs.append(emb)

            embs = torch.cat(all_embs, dim=0)
            if embs.size(0) != global_n_seqs:
                raise DiversityGroupingError(
                    f"Re-forward produced {embs.size(0)} embeddings, "
                    f"expected {global_n_seqs}"
                )
            embs_norm = F.normalize(embs, dim=-1)
            all_structure_indices = all_structure_indices_cpu.to(device)

            # Compute within-SID off-diagonal cosine means and assign equal weight
            # to each SID group in the generation-round regularizer.
            group_mean_sims = []
            for structure_index_value in unique_indices_cpu.tolist():
                group_mask = all_structure_indices.eq(structure_index_value)
                group_embs = embs_norm[group_mask]
                group_size = group_embs.size(0)
                if group_size != num_generations:
                    raise DiversityGroupingError(
                        f"SID index {structure_index_value} has {group_size} "
                        f"embeddings, expected {num_generations}"
                    )
                sim_matrix = group_embs @ group_embs.t()
                mask_diag = 1.0 - torch.eye(group_size, device=device)
                group_mean_sims.append(
                    (sim_matrix * mask_diag).sum() / mask_diag.sum()
                )
            mean_sim = torch.stack(group_mean_sims).mean()

            spg = self.args.steps_per_generation
            normalizer = self.current_gradient_accumulation_steps / spg
            div_loss = self.alpha_diversity * mean_sim / normalizer
            loss = loss + div_loss

            self._metrics["train"]["diversity"].append((1.0 - mean_sim).item())
            self._metrics["train"]["diversity_loss"].append(div_loss.item())
            self._metrics["train"]["grpo_loss"].append(
                loss.item() - div_loss.item()
            )

        except DiversityGroupingError:
            raise
        except Exception as e:
            self._div_input_buffer.clear()
            logger.warning(f"Pairwise diversity penalty failed (skipping): {e}")

        return loss

    # Multi-adapter PEFT checkpoint handling
    #
    # PEFT stores the trainable `default` adapter at the checkpoint root and
    # the KL `ref` adapter in a subdirectory. The base Trainer may then skip
    # the root adapter, so these overrides save and restore it explicitly.

    def _policy_contract_manifest(self):
        spec = self.protein_action_space
        return {
            "schema": "proteinzero_grpo_policy_contract_v1",
            "source": {
                "train_source_sha256": sha256_file(Path(__file__).resolve()),
            },
            "versions": {
                "python": ".".join(map(str, sys.version_info[:3])),
                "torch": torch.__version__,
                **{name: package_version(name) for name in EXPECTED_POLICY_VERSIONS
                   if name != "torch"},
            },
            "action_space": {
                "vocab_size": spec.vocab_size,
                "aa_token_ids": list(spec.aa_token_ids),
                "sentinel1_id": spec.sentinel1_id,
                "sentinel2_id": spec.sentinel2_id,
                "special_eos_id": spec.special_eos_id,
                "pad_id": spec.pad_id,
                "sampling": {
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "top_k": self.args.top_k,
                    "min_p": self.args.min_p,
                    "repetition_penalty": self.args.repetition_penalty,
                    "force_exact_length": self.force_exact_length,
                    "use_vllm": self.use_vllm,
                },
            },
            "precision_dropout_contract": {
                "disable_dropout": self.args.disable_dropout,
                "bf16": self.args.bf16,
                "fp16": self.args.fp16,
                "bf16_full_eval": self.args.bf16_full_eval,
                "fp16_full_eval": self.args.fp16_full_eval,
                "tf32": self.args.tf32,
                "derived_mixed_precision": getattr(self.args, "mixed_precision", None),
                "accelerator_mixed_precision": self.accelerator.mixed_precision,
                "accelerator_native_amp": self.accelerator.native_amp,
                "accelerator_scaler_is_none": self.accelerator.scaler is None,
                "parameter_dtype": "torch.float32",
            },
            "policy_provenance": copy.deepcopy(self.policy_provenance),
            "training_observations": {
                "num_iterations": self.args.num_iterations,
                "loss_type": self.args.loss_type,
                "importance_sampling_level": self.args.importance_sampling_level,
                "use_transformers_paged": getattr(
                    self.args, "use_transformers_paged", False
                ),
                "use_liger_kernel": self.args.use_liger_kernel,
            },
        }

    def _mpnn_gen_state_file(self, directory):
        if self.args.world_size <= 1:
            name = "mpnn_gen_state.pth"
        else:
            name = f"mpnn_gen_state_{self.args.process_index}.pth"
        return os.path.join(directory, name)

    def _save_rng_state(self, output_dir):
        """Save HF RNG state plus this rank's stateful ddG generator.

        Per-rank storage matches HF's RNG checkpoint convention and preserves
        any rank-local divergence in fast_ddg call history.
        """
        super()._save_rng_state(output_dir)
        if mpnn_gen is None:
            logger.warning("mpnn_gen not initialized; skipping ddG RNG checkpoint")
            return
        torch.save(mpnn_gen.get_state(), self._mpnn_gen_state_file(output_dir))

    def _load_rng_state(self, checkpoint):
        """Restore HF RNG state plus this rank's stateful ddG generator.

        The per-rank file restores the generator state saved for this process.
        """
        super()._load_rng_state(checkpoint)
        if checkpoint is None:
            return
        if mpnn_gen is None:
            logger.warning("mpnn_gen not initialized; skipping ddG RNG restore")
            return
        path = self._mpnn_gen_state_file(checkpoint)
        if not os.path.isfile(path):
            logger.warning(
                "No rank %d ddG generator state at %s; keeping the freshly "
                "seeded generator. Exact ddG reproducibility across this "
                "resume is not guaranteed.",
                self.args.process_index,
                path,
            )
            return
        mpnn_gen.set_state(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        logger.info(f"Restored ddG generator state from {path}")

    def _verify_resume_contract(self, resume_from_checkpoint):
        allow_incompatible_resume = "allow_incompatible_resume" in str(self.args.output_dir)
        manifest_path = Path(resume_from_checkpoint) / "policy_contract.json"
        if allow_incompatible_resume:
            logger.warning(
                "Incompatible checkpoint continuation explicitly enabled; "
                "policy-contract verification is skipped"
            )
            return
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Clean resume requires policy contract manifest: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            actual = json.load(handle)
        expected = self._policy_contract_manifest()

        hard_fields = (
            "schema",
            "action_space",
            "precision_dropout_contract",
            "policy_provenance",
        )
        differing = [
            field for field in hard_fields
            if actual.get(field) != expected.get(field)
        ]
        if differing:
            raise RuntimeError(
                f"Resume policy contract mismatch in hard fields: {differing}"
            )

        actual_source = actual.get("source", {}).get(
            "train_source_sha256", actual.get("train_source_sha256")
        )
        expected_source = expected["source"]["train_source_sha256"]
        if actual_source != expected_source:
            logger.warning(
                "Checkpoint source SHA256 differs (record-only): "
                f"checkpoint={actual_source}, runtime={expected_source}"
            )
        for field in ("versions", "training_observations"):
            if actual.get(field) != expected.get(field):
                logger.warning(
                    f"Resume {field} differs (record-only): "
                    f"checkpoint={actual.get(field)}, runtime={expected.get(field)}"
                )
        logger.info(f"Verified clean resume hard contract: {manifest_path}")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Load PEFT adapter weights from a checkpoint, handling both
        root-level ("default") and subdirectory ("ref", etc.) adapters.

        Fixes the base Trainer bug where the presence of a ref/ subdirectory
        causes the root-level "default" adapter to be skipped entirely.
        """
        if model is None:
            model = self.model
        self._verify_resume_contract(resume_from_checkpoint)

        from peft import PeftModel as _PeftModel

        unwrapped = self.accelerator.unwrap_model(model)
        if not isinstance(unwrapped, _PeftModel):
            result = super()._load_from_checkpoint(resume_from_checkpoint, model)
            self._enforce_and_audit_policy_runtime("resume_non_peft", require_no_amp_wrapper=True)
            return result

        if not os.path.isdir(resume_from_checkpoint):
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {resume_from_checkpoint}"
            )

        active_adapter = unwrapped.active_adapters[0]
        logger.info(f"[ProteinGRPOTrainer] Loading PEFT checkpoint from {resume_from_checkpoint}")
        logger.info(f"  Active adapter: '{active_adapter}'")

        root_safe = os.path.join(resume_from_checkpoint, "adapter_model.safetensors")
        root_bin = os.path.join(resume_from_checkpoint, "adapter_model.bin")
        if os.path.isfile(root_safe) or os.path.isfile(root_bin):
            load_result = unwrapped.load_adapter(
                resume_from_checkpoint, active_adapter, is_trainable=True,
            )
            if load_result.missing_keys or load_result.unexpected_keys:
                raise RuntimeError(
                    f"Incomplete adapter resume for {active_adapter}: "
                    f"missing={load_result.missing_keys}, "
                    f"unexpected={load_result.unexpected_keys}"
                )
            logger.info(f"  Loaded adapter '{active_adapter}' from checkpoint root")
        else:
            raise FileNotFoundError(
                f"No adapter weights at checkpoint root: {resume_from_checkpoint}"
            )

        for name in sorted(os.listdir(resume_from_checkpoint)):
            subdir = os.path.join(resume_from_checkpoint, name)
            if not os.path.isdir(subdir):
                continue
            has_weights = (
                os.path.isfile(os.path.join(subdir, "adapter_model.safetensors"))
                or os.path.isfile(os.path.join(subdir, "adapter_model.bin"))
            )
            if not has_weights:
                continue
            if name not in unwrapped.peft_config:
                raise RuntimeError(f"Unknown checkpoint adapter subdirectory: {name}")
            load_result = unwrapped.load_adapter(
                subdir, name, is_trainable=(name == active_adapter),
            )
            if load_result.missing_keys or load_result.unexpected_keys:
                raise RuntimeError(
                    f"Incomplete adapter resume for {name}: "
                    f"missing={load_result.missing_keys}, "
                    f"unexpected={load_result.unexpected_keys}"
                )
            logger.info(f"  Loaded adapter '{name}' from subdirectory")

        unwrapped.set_adapter(active_adapter)
        logger.info(f"  Active adapter set to '{active_adapter}'")
        self._enforce_and_audit_policy_runtime("resume_peft", require_no_amp_wrapper=True)

    def _save(self, output_dir=None, state_dict=None):
        """Save only the 'default' adapter to avoid creating a ref/ subdirectory.

        The 'ref' adapter is recreated from 'default' by GRPOTrainer.__init__
        on every launch, so persisting it is unnecessary and causes a loading
        bug in the base Trainer (see _load_from_checkpoint above).
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(
            output_dir,
            selected_adapters=["default"],
            safe_serialization=getattr(self.args, "save_safetensors", True),
        )
        logger.info(f"  Saved adapter 'default' only (skipped 'ref')")

        manifest_path = Path(output_dir) / "policy_contract.json"
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        with manifest_tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                self._policy_contract_manifest(), handle, indent=2, sort_keys=True
            )
            handle.write("\n")
        os.replace(manifest_tmp, manifest_path)
        logger.info(f"  Saved policy contract: {manifest_path}")

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _prepare_inputs(self, generation_batch):
        if not self._strict_runtime_checked:
            self._enforce_and_audit_policy_runtime("post_prepare", require_no_amp_wrapper=True)
            self._strict_runtime_checked = True
        inputs = super()._prepare_inputs(generation_batch)
        if self.model.training:
            spg = self.args.steps_per_generation
            # TRL 0.27.2 grpo_trainer.py:1158-1161 increments _step only
            # after training_step returns; :1184-1194 selects this buffered
            # slice with the same, not-yet-incremented zero-based _step.
            self._is_new_backbone = self._step % spg == 0
            self._is_last_micro_of_backbone = (self._step + 1) % spg == 0
        return inputs



# LoRA configuration

def discover_lora_target_modules(model) -> list:
    """
    Discover LoRA target modules for ProGen2 architecture.
    Scans for attention and MLP layers.
    """
    target_patterns = [
        "query_key_value",
        "attn.q", "attn.out",
        "mlp.fc_in", "mlp.fc_out",
        "q_proj", "k_proj", "v_proj", "out_proj",
    ]
    target_modules = []
    for name, module in model.named_modules():
        if any(pattern in name for pattern in target_patterns):
            if hasattr(module, "weight") and module.weight is not None:
                target_modules.append(name)

    if not target_modules:
        logger.warning("No target modules found! Falling back to 'all-linear'.")
        return "all-linear"

    logger.info(f"Found {len(target_modules)} LoRA target modules.")
    return target_modules


def create_lora_config(model, args: ProteinGRPOArguments) -> LoraConfig:
    """Create LoRA configuration for the ProGen2 model."""
    target_modules = discover_lora_target_modules(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    return lora_config


# Exact-length generation support

def init_force_exact_length(tokenizer, model_config):
    """Initialize the strict shared protein action space."""
    global PROTEIN_ACTION_SPACE
    spec = build_protein_action_space(tokenizer, model_config)
    if PROTEIN_ACTION_SPACE is not None and PROTEIN_ACTION_SPACE != spec:
        raise RuntimeError("Attempted to replace the immutable ProteinActionSpace")
    PROTEIN_ACTION_SPACE = spec
    return spec


def build_sid_to_target_info(train_list, val_list=None):
    """
    Build mapping from structure_id to target sequence length.
    Used by ForceExactLength to determine the correct length per backbone.
    """
    global SID_TO_TARGET_INFO
    SID_TO_TARGET_INFO = {}
    for item_list in [train_list, val_list or []]:
        for item in item_list:
            sid = item["structure_id"]
            native_seq = item["native_seq"]
            if sid not in SID_TO_TARGET_INFO:
                SID_TO_TARGET_INFO[sid] = {
                    "target_len": len(native_seq),
                }
    logger.info(f"Built target info for {len(SID_TO_TARGET_INFO)} structures.")
    return SID_TO_TARGET_INFO


# Dataset preparation

def prepare_dataset(train_list: list) -> Dataset:
    """
    Convert the training list into a HuggingFace Dataset for GRPOTrainer.

    The dataset must have a 'prompt' column. All other columns are passed
    as kwargs to reward functions automatically.

    Prompt format: "{structure_id}|1"
    The model generates amino acid tokens followed by sentinel "2".
    """
    records = []
    for item in train_list:
        sid = item["structure_id"]
        native_seq = item["native_seq"]
        records.append({
            "prompt": f"{sid}|1",                 # Structure identifier followed by the initial sentinel.
            "structure_id": sid,                   # Passed to the reward functions.
            "native_seq": native_seq,              # Passed to the recovery reward.
            "native_seq_len": len(native_seq),     # Passed to the length reward.
        })

    dataset = Dataset.from_list(records)
    logger.info(f"Prepared dataset with {len(dataset)} structures.")
    logger.info(f"  Sequence lengths: min={min(r['native_seq_len'] for r in records)}, "
                f"max={max(r['native_seq_len'] for r in records)}, "
                f"mean={np.mean([r['native_seq_len'] for r in records]):.1f}")
    return dataset


# Validation

def evaluate_all_metrics(
    model,
    tokenizer,
    val_list: list,
    pdb_val_dir: str,
    device: torch.device,
    num_samples: int = 1,
    max_structures: int = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    force_exact_length: bool = True,
    output_dir: str = None,
    step: int = 0,
    reward_weights: list = None,
    protein_action_space: ProteinActionSpace = None,
) -> dict:
    """
    Run distributed validation by assigning disjoint backbone shards to ranks.
    Each rank processes its assigned backbones sequentially, and aggregate
    metrics are merged once after all local work is complete.

    Candidates are selected per structure by the configured combined reward
    using the same group normalization as training. Selection falls back to
    pLDDT when no objective weights are active.

    When force_exact_length=True, uses ForceExactLength logits processor
    to enforce the native backbone length.

    Returns dict of aggregated metrics.
    """
    spec = protein_action_space if protein_action_space is not None else PROTEIN_ACTION_SPACE
    if not force_exact_length or spec is None:
        raise RuntimeError("Validation requires the shared exact protein action space")
    if num_samples < 1:
        raise ValueError(f"Validation num_samples must be positive, got {num_samples}")
    model.eval()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    gen_model = model.module if hasattr(model, "module") else model

    if max_structures is None:
        max_structures = len(val_list)
    subset = val_list[:max_structures]
    local_subset = subset[rank::world_size]
    shard_sizes = [len(subset[r::world_size]) for r in range(world_size)]

    if rank == 0:
        logger.info(
            "Validation schedule: %d backbones, %d sequence(s) per backbone, "
            "%d rank(s), rank shards=%s",
            len(subset), num_samples, world_size, shard_sizes,
        )

    # Accumulators
    local_rr_sum = 0.0
    local_tm_sum = 0.0
    local_scrmsd_sum = 0.0
    local_plddt_sum = 0.0
    local_ddg_sum = 0.0
    local_reward_sum = 0.0
    local_lt2_cnt = 0
    local_n = 0
    local_div_sum = 0.0
    local_div_struct_cnt = 0
    total_count = 0
    valid_count = 0

    for item in tqdm(local_subset, desc=f"Validation rank {rank}", disable=(rank != 0)):
        sid = item["structure_id"]
        native_seq = item["native_seq"]
        target_len = len(native_seq)

        prompt_str = f"{sid}|1"
        enc = tokenizer(prompt_str, return_tensors="pt").to(device)

        enc["attention_mask"] = torch.ones_like(enc["attention_mask"])

        logits_proc = LogitsProcessorList([
            ForceExactLength(spec, target_len, prefix_len=None)
        ])

        local_seq_metrics = []
        try:
            val_eos_ids = [
                spec.special_eos_id,
                spec.sentinel2_id,
            ]

            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                outputs = gen_model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=target_len + 20,
                    min_new_tokens=(target_len + 1) if force_exact_length else 0,
                    do_sample=True,
                    temperature=temperature,
                    top_k=0,
                    top_p=top_p,
                    num_return_sequences=num_samples,
                    logits_processor=logits_proc,
                    eos_token_id=val_eos_ids,
                )

            for output_ids in outputs:
                full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                if "|1" in full_text:
                    completion = full_text.split("|1", 1)[1]
                else:
                    completion = full_text

                seq = clean(completion)
                total_count += 1

                if not seq or len(seq) < 10:
                    continue
                valid_count += 1

                rr = partial_recovery(seq, native_seq)
                plddt = fast_plddt(seq)
                tm = fast_tm_score(seq, sid, pdb_dir=pdb_val_dir)
                scrmsd = fast_sc_rmsd(seq, sid, pdb_dir=pdb_val_dir)
                ddg = fast_ddg(seq, sid)
                local_seq_metrics.append((seq, rr, tm, scrmsd, plddt, ddg))

        except Exception as e:
            logger.warning(f"Validation failed for {sid}: {e}")
        finally:
            torch.cuda.empty_cache()

        # A complete candidate group belongs to one rank. No per-backbone
        # collective is required.
        all_seq_metrics = local_seq_metrics

        if not all_seq_metrics:
            continue

        top_n = max(1, int(len(all_seq_metrics) * TOP_RATIO))
        combined_scores = _compute_combined_reward_for_group(
            all_seq_metrics, reward_weights
        )
        scored = list(zip(combined_scores, all_seq_metrics))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_half_scores = [(s, m) for s, m in scored[:top_n]]

        for reward_score, (seq, rr, tm, scrmsd, plddt, ddg) in top_half_scores:
            local_rr_sum += rr
            local_tm_sum += tm
            local_scrmsd_sum += scrmsd
            local_plddt_sum += plddt
            local_ddg_sum += ddg
            local_reward_sum += reward_score
            local_n += 1
            if scrmsd < 2.0:
                local_lt2_cnt += 1

        top_seqs = [m[0] for _, m in top_half_scores]
        if len(top_seqs) >= 2:
            div_val = seq_diversity(top_seqs)
            local_div_sum += div_val
            local_div_struct_cnt += 1

    local_totals = {
        "rr_sum": local_rr_sum,
        "tm_sum": local_tm_sum,
        "scrmsd_sum": local_scrmsd_sum,
        "plddt_sum": local_plddt_sum,
        "ddg_sum": local_ddg_sum,
        "reward_sum": local_reward_sum,
        "lt2_cnt": local_lt2_cnt,
        "n": local_n,
        "div_sum": local_div_sum,
        "div_struct_cnt": local_div_struct_cnt,
        "total_count": total_count,
        "valid_count": valid_count,
    }

    if dist.is_initialized() and world_size > 1:
        gathered_totals = [None] * world_size
        dist.all_gather_object(gathered_totals, local_totals)
    else:
        gathered_totals = [local_totals]

    totals = {
        key: sum(rank_totals[key] for rank_totals in gathered_totals)
        for key in local_totals
    }
    global_n = int(totals["n"])

    if global_n == 0:
        if rank == 0:
            logger.warning("No valid sequences in validation!")
        model.train()
        return {"val/tm_scores_mean": 0.0, "val/plddts_mean": 0.0,
                "val/recoveries_mean": 0.0, "val/sc_rmsds_mean": 0.0}

    div_struct_count = int(totals["div_struct_cnt"])
    avg_div = totals["div_sum"] / div_struct_count if div_struct_count > 0 else 0.0

    metrics = {
        "val/tm_scores_mean": totals["tm_sum"] / global_n,
        "val/plddts_mean": totals["plddt_sum"] / global_n,
        "val/recoveries_mean": totals["rr_sum"] / global_n,
        "val/sc_rmsds_mean": totals["scrmsd_sum"] / global_n,
        "val/ddg_mean": totals["ddg_sum"] / global_n,
        "val/frac_scrmsd_lt2": totals["lt2_cnt"] / global_n,
        "val/diversity": avg_div,
        "val/valid_ratio": totals["valid_count"] / max(totals["total_count"], 1),
        "val/total_evaluated": float(totals["total_count"]),
    }

    if rank == 0:
        logger.info("=" * 60)
        ranking_desc = "combined reward" if reward_weights and any(w != 0.0 for _, w in reward_weights) else "pLDDT"
        logger.info(f"Validation Results (top-{int(TOP_RATIO*100)}% {ranking_desc}, {global_n} seqs from {len(subset)} structures, {world_size} GPUs):")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info("=" * 60)

        if output_dir:
            val_log_path = os.path.join(output_dir, "val_metrics.txt")
            write_header = not os.path.exists(val_log_path)
            with open(val_log_path, "a") as f:
                if write_header:
                    f.write("step\trr\ttm\tscrmsd\tfrac_lt2\tplddt\tddg\treward\tdiversity\n")
                f.write(f"{step}\t{metrics['val/recoveries_mean']:.4f}\t"
                        f"{metrics['val/tm_scores_mean']:.4f}\t"
                        f"{metrics['val/sc_rmsds_mean']:.4f}\t"
                        f"{metrics['val/frac_scrmsd_lt2']:.4f}\t"
                        f"{metrics['val/plddts_mean']:.2f}\t"
                        f"{metrics['val/ddg_mean']:.4f}\t"
                        f"{totals['reward_sum'] / global_n:.4f}\t"
                        f"{avg_div:.4f}\n")

    model.train()
    return metrics


# Validation callback

class ProteinValidationCallback(TrainerCallback):
    """
    Custom callback to run protein design validation during GRPO training.
    Backbones are sharded across ranks, and each rank processes its shard
    sequentially.

    The fixed validation interval is configured by validate_every_steps. All
    ranks participate; rank zero records the merged metrics.
    """
    def __init__(self, val_list, pdb_val_dir, tokenizer, device,
                 validate_every_steps=0, force_exact_length=True,
                 output_dir=None, num_samples=1,
                 reward_weights=None, protein_action_space=None,
                 eval_temperature: float = 0.7, eval_top_p: float = 1.0):
        self.val_list = val_list
        self.pdb_val_dir = pdb_val_dir
        self.tokenizer = tokenizer
        self.device = device
        self.validate_every_steps = validate_every_steps
        self.force_exact_length = force_exact_length
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.reward_weights = reward_weights
        self.protein_action_space = protein_action_space
        self.eval_temperature = eval_temperature
        self.eval_top_p = eval_top_p

    def on_train_begin(self, args, state, control, **kwargs):
        """Align checkpoint cadence with the configured validation interval."""
        if self.validate_every_steps > 0:
            args.save_steps = self.validate_every_steps
            logger.info(f"Validate every {self.validate_every_steps} steps (from --validate_every_steps)")
            logger.info(f"  save_steps = {args.save_steps}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.validate_every_steps > 0 and state.global_step % self.validate_every_steps == 0:
            if state.global_step == 0:
                return  # Skip step 0
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                logger.info(f"\n[Step {state.global_step}] Running validation (distributed across all GPUs)...")
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            metrics = evaluate_all_metrics(
                model=model,
                tokenizer=self.tokenizer,
                val_list=self.val_list,
                pdb_val_dir=self.pdb_val_dir,
                device=self.device,
                num_samples=self.num_samples,
                max_structures=len(self.val_list),
                temperature=self.eval_temperature,
                top_p=self.eval_top_p,
                force_exact_length=self.force_exact_length,
                output_dir=self.output_dir,
                step=state.global_step,
                reward_weights=self.reward_weights,
                protein_action_space=self.protein_action_space,
            )
            if local_rank == 0 and hasattr(state, "log_history"):
                state.log_history.append(metrics)


class TrainMetricsLoggingCallback(TrainerCallback):
    """
    Periodically flush accumulated training metrics (top-{TOP_RATIO*100}% by combined reward) to txt files.
    Runs on rank zero and writes the accumulated training diagnostics.
    """
    def __init__(self, output_dir: str, flush_every_steps: int = 10):
        self.output_dir = output_dir
        self.flush_every_steps = flush_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank != 0:
            return
        if state.global_step > 0 and state.global_step % self.flush_every_steps == 0:
            _flush_train_metrics(step=state.global_step, output_dir=self.output_dir)


# Training entry point

def main():
    global ESMFOLD_MODEL, TOKENIZER, DEVICE
    global PDB_TRAIN_DIR, TRAIN_LIST, VAL_LIST, PROTEIN_ACTION_SPACE

    # Set before GRPOConfig/TrainingArguments and Accelerator are instantiated.
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

    # Configuration
    parser = HfArgumentParser((ProteinGRPOArguments, GRPOConfig))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        protein_args, grpo_config = parser.parse_json_file(sys.argv[1])
    else:
        protein_args, grpo_config = parser.parse_args_into_dataclasses()

    _cv = getattr(protein_args, "cath_version", "4.3") or "4.3"
    _cath_dir = f"cath{_cv.replace('.', '')}"
    if _cv == "4.3":
        if protein_args.pdb_train_dir is None:
            protein_args.pdb_train_dir = f"{_cath_dir}/pdbs/train"
        if protein_args.pdb_val_dir is None:
            protein_args.pdb_val_dir = f"{_cath_dir}/pdbs/val"
        if protein_args.train_csv is None:
            protein_args.train_csv = f"{_cath_dir}/train_data_le100.csv"
        if protein_args.val_csv is None:
            protein_args.val_csv = f"{_cath_dir}/val_data_le100.csv"
        if protein_args.structure_emb_path_prefix is None:
            protein_args.structure_emb_path_prefix = f"./structure_embeddings_{_cath_dir}"

    grpo_config.temperature = protein_args.gen_temperature
    grpo_config.top_p = protein_args.gen_top_p
    grpo_config.beta = protein_args.kl_beta
    configure_strict_lm_policy_runtime(grpo_config, protein_args)

    # Use non-reentrant gradient checkpointing with DDP.
    if grpo_config.gradient_checkpointing:
        grpo_config.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # The command-line parser assigns loss_type, epsilon, epsilon_high,
    # num_generations, and seed directly to grpo_config.

    if grpo_config.output_dir == "tmp_trainer":
        grpo_config.output_dir = "./grpo_protein_output"

    set_seed(grpo_config.seed)

    # Distributed runtime
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    DEVICE = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if local_rank != 0:
        logger.setLevel(logging.WARNING)
    logger.info(f"Using device: {DEVICE}")

    # Data
    PDB_TRAIN_DIR = protein_args.pdb_train_dir

    logger.info(f"Loading training data from {protein_args.train_csv}...")
    TRAIN_LIST = []
    with open(protein_args.train_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["structure_id"].strip()
            if not sid:
                continue
            TRAIN_LIST.append({
                "structure_id": sid,
                "native_seq": row["seq"],
            })
    logger.info(f"Loaded {len(TRAIN_LIST)} training structures.")

    val_list = []
    if os.path.exists(protein_args.val_csv):
        with open(protein_args.val_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["structure_id"].strip()
                if not sid:
                    continue
                val_list.append({
                    "structure_id": sid,
                    "native_seq": row["seq"],
                })
        logger.info(f"Loaded {len(val_list)} validation structures.")
    VAL_LIST = val_list

    all_seq_lens = [len(item["native_seq"]) for item in TRAIN_LIST]
    max_seq_len = max(all_seq_lens)
    # Reserve completion capacity for the terminal sentinel and tokenizer overhead.
    max_completion_length = max_seq_len + 30
    grpo_config.max_completion_length = max_completion_length
    logger.info(f"Max sequence length: {max_seq_len}, max_completion_length: {max_completion_length}")

    # Set prompt capacity for 256 structure positions and the start sentinel.
    if not hasattr(grpo_config, 'max_prompt_length') or grpo_config.max_prompt_length is None:
        grpo_config.max_prompt_length = 300  # Includes tokenizer overhead.

    # ProGen2 batching constraint
    # The structure encoder requires structure tokens at position zero.
    # Per-device batches larger than one introduce left padding, so the batch
    # dimension is transferred to gradient accumulation.
    if grpo_config.per_device_train_batch_size > 1:
        logger.warning(
            f"per_device_train_batch_size={grpo_config.per_device_train_batch_size} > 1. "
            f"ProGen2's structure encoder requires batch_size=1 to avoid left-padding issues. "
            f"Forcing per_device_train_batch_size=1 and adjusting gradient_accumulation_steps."
        )
        orig_bs = grpo_config.per_device_train_batch_size
        grpo_config.gradient_accumulation_steps = (
            grpo_config.gradient_accumulation_steps * orig_bs
        )
        grpo_config.per_device_train_batch_size = 1
        logger.info(
            f"Adjusted batch_size=1, gradient_accumulation_steps={grpo_config.gradient_accumulation_steps}"
        )

    # Policy model
    logger.info(f"Loading model from {protein_args.model_name}...")
    config = AutoConfig.from_pretrained(protein_args.model_name, trust_remote_code=True)
    if hasattr(config, "structure") and isinstance(config.structure, dict):
        config.structure["structure_emb_path_prefix"] = protein_args.structure_emb_path_prefix
        logger.info(f"Structure embedding prefix: {protein_args.structure_emb_path_prefix}")
    base_model = AutoModelForCausalLM.from_pretrained(
        protein_args.model_name,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.float32,  # ProGen2 execution uses FP32.
    )

    # Transformers 5.x does not reliably populate this custom architecture;
    # load every checkpoint tensor explicitly and verify complete coverage.
    from safetensors import safe_open
    _ckpt_dir = protein_args.model_name
    _shard_files = sorted([
        f for f in os.listdir(_ckpt_dir)
        if f.endswith('.safetensors') and 'model' in f
    ])
    if _shard_files:
        _full_sd = {}
        for _sf in _shard_files:
            with safe_open(os.path.join(_ckpt_dir, _sf), framework="pt", device="cpu") as _f:
                for _key in _f.keys():
                    if _key in _full_sd:
                        raise RuntimeError(f"Duplicate checkpoint tensor: {_key}")
                    _full_sd[_key] = _f.get_tensor(_key)

        _model_sd = base_model.state_dict()
        if set(_full_sd) != set(_model_sd):
            raise RuntimeError(
                f"Checkpoint/model key mismatch: "
                f"missing={sorted(set(_model_sd) - set(_full_sd))[:10]}, "
                f"unexpected={sorted(set(_full_sd) - set(_model_sd))[:10]}"
            )
        _loaded = 0
        for _key, _tensor in _full_sd.items():
            if _key in _model_sd and _model_sd[_key].shape == _tensor.shape:
                _model_sd[_key].copy_(_tensor)
                _loaded += 1

        _test_param = dict(base_model.named_parameters())["transformer.h.0.ln_1.weight"]
        if abs(_test_param.data.mean().item() - 1.0) < 0.001:
            raise RuntimeError(
                "Manual weight loading failed! transformer.h.0.ln_1.weight is still "
                "at default init (mean=1.0). Check checkpoint files."
            )
        if len(_full_sd) != 339 or _loaded != 339:
            raise RuntimeError(
                f"Expected production checkpoint load 339/339, got {_loaded}/{len(_full_sd)}"
            )
        logger.info("Manually loaded 339/339 model checkpoint tensors")
        del _full_sd  # Release the checkpoint state dictionary.
    else:
        raise RuntimeError("No production safetensors shards found")

    TOKENIZER = AutoTokenizer.from_pretrained(
        protein_args.model_name,
        trust_remote_code=True,
    )
    # ProteinActionSpace below normalizes raw tokenizer EOS from the model
    # config. The remaining special tokens are likewise pinned by model assets.
    if TOKENIZER.bos_token_id is None:
        TOKENIZER.bos_token_id = base_model.config.bos_token_id  # 1
    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token_id = 0  # the model uses 0 for padding

    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in base_model.parameters()):,}")

    PROTEIN_ACTION_SPACE = init_force_exact_length(TOKENIZER, base_model.config)
    build_sid_to_target_info(TRAIN_LIST, val_list)
    logger.info("ForceExactLength: required; A20 rollout/replay contract enabled")

    # Frozen stability model
    # GRPOTrainer manages the KL reference internally; physical ΔΔG is scored
    # by the frozen ProteinMPNN model.
    global mpnn_model, mpnn_gen, mpnn_feats_cache, native_seqs
    logger.info("Loading ProteinMPNN for ΔΔG scoring...")
    mpnn_model, mpnn_gen = load_mpnn_model(DEVICE, grpo_config.seed)
    logger.info(f"  ProteinMPNN v_48_020 loaded (seed={grpo_config.seed})")

    mpnn_feats_cache = {}
    native_seqs = {}
    n_mpnn_ok, n_mpnn_fail = 0, 0
    all_items = TRAIN_LIST + (VAL_LIST or [])
    for idx, item in enumerate(all_items):
        sid = item["structure_id"]
        if sid in mpnn_feats_cache:
            continue
        pdb_path = os.path.join(PDB_TRAIN_DIR, f"{sid}.pdb")
        if not os.path.isfile(pdb_path):
            pdb_val_dir = protein_args.pdb_val_dir if hasattr(protein_args, 'pdb_val_dir') else "./pdbs/val"
            pdb_path = os.path.join(pdb_val_dir, f"{sid}.pdb")
        if not os.path.isfile(pdb_path):
            n_mpnn_fail += 1
            continue
        fc, fu = prepare_mpnn_feats(pdb_path, DEVICE)
        if fc is None:
            n_mpnn_fail += 1
            continue
        mpnn_feats_cache[sid] = (fc, fu)
        native_seqs[sid] = clean(item["native_seq"])
        n_mpnn_ok += 1
        if n_mpnn_ok % 500 == 0:
            logger.info(f"    Pre-computed MPNN features: {n_mpnn_ok}/{len(all_items)}")
    logger.info(f"  MPNN features ready: {n_mpnn_ok} ok, {n_mpnn_fail} failed")

    # Structure predictor
    init_esmfold(DEVICE)

    # LoRA policy
    lora_config = create_lora_config(base_model, protein_args)
    logger.info(f"LoRA config: rank={lora_config.r}, alpha={lora_config.lora_alpha}")

    _policy_provenance = {"mode": "fresh_lora"}

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if dist.is_initialized():
        dist.barrier()
    if local_rank == 0:
        gpu_names = []
        for i in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(i))
        logger.info("=" * 60)
        logger.info(f"All {world_size} GPUs initialized successfully.")
        logger.info(f"  GPUs: {gpu_names[0]} x {world_size}")
        logger.info(f"  Model: {sum(p.numel() for p in base_model.parameters()):,} params")
        logger.info(f"  MPNN features: {n_mpnn_ok} structures cached")
        logger.info(f"  Training set: {len(TRAIN_LIST)} structures")
        logger.info(f"  Validation set: {len(VAL_LIST)} structures")
        logger.info("=" * 60)

    train_dataset = prepare_dataset(TRAIN_LIST)

    # Reward configuration
    val_reward_weights = [
        ("tm", protein_args.reward_weight_tm),
        ("ddg", protein_args.reward_weight_ddg),
        ("plddt", protein_args.reward_weight_plddt),
        ("recovery", protein_args.reward_weight_recovery),
        ("length", protein_args.reward_weight_length),
    ]
    global REWARD_WEIGHTS_GLOBAL
    REWARD_WEIGHTS_GLOBAL = [
        (name, weight)
        for name, weight in val_reward_weights
        if weight != 0.0
    ]

    reward_funcs = [multi_objective_reward]
    grpo_config.reward_weights = [1.0]

    logger.info(
        "Reward: multi_objective_reward (paper Sec 3.1.2, linear "
        f"min-max weighted): {dict(REWARD_WEIGHTS_GLOBAL)}"
    )
    # Output and callbacks.
    # Diagnostics are stored at the run root; checkpoints use its checkpoints subdirectory.
    log_dir = grpo_config.output_dir  # Diagnostic output root.
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    grpo_config.output_dir = ckpt_dir  # Trainer checkpoint directory.
    grpo_config.save_total_limit = None  # Retain all checkpoints.
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    active_rw = [
        (name, weight)
        for name, weight in val_reward_weights
        if weight != 0.0
    ]
    logger.info(f"Validation top-{int(TOP_RATIO*100)}% ranking by combined reward: {active_rw}")

    callbacks = []
    if protein_args.validate_every_steps > 0 and val_list:
        callbacks.append(ProteinValidationCallback(
            val_list=val_list,
            pdb_val_dir=protein_args.pdb_val_dir,
            tokenizer=TOKENIZER,
            device=DEVICE,
            validate_every_steps=protein_args.validate_every_steps,
            force_exact_length=protein_args.force_exact_length,
            output_dir=log_dir,  # Diagnostic output root.
            num_samples=protein_args.val_num_generations,
            reward_weights=val_reward_weights,
            protein_action_space=PROTEIN_ACTION_SPACE,
            eval_temperature=protein_args.eval_temperature,
            eval_top_p=protein_args.eval_top_p,
        ))

    # Training diagnostics
    callbacks.append(TrainMetricsLoggingCallback(
        output_dir=log_dir,  # Diagnostic output root.
        flush_every_steps=grpo_config.logging_steps,
    ))

    # Trainer
    logger.info("Creating ProteinGRPOTrainer...")
    logger.info(f"  loss_type: {grpo_config.loss_type}")
    logger.info(f"  beta (KL): {grpo_config.beta}")
    logger.info(f"  epsilon: {grpo_config.epsilon}")
    logger.info(f"  num_generations: {grpo_config.num_generations}")
    logger.info(f"  max_completion_length: {grpo_config.max_completion_length}")
    logger.info(f"  temperature (GRPO rollouts): {grpo_config.temperature}")
    logger.info(f"  top_p (GRPO rollouts): {grpo_config.top_p}")
    logger.info(f"  eval_temperature (validation): {protein_args.eval_temperature}")
    logger.info(f"  eval_top_p (validation): {protein_args.eval_top_p}")
    logger.info(f"  val_num_generations (validation): {protein_args.val_num_generations}")
    logger.info(f"  alpha_diversity: {protein_args.alpha_diversity}")
    logger.info(f"  force_exact_length: {protein_args.force_exact_length}")

    trainer = ProteinGRPOTrainer(
        model=base_model,
        processing_class=TOKENIZER,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        peft_config=lora_config,
        callbacks=callbacks,
        alpha_diversity=protein_args.alpha_diversity,
        force_exact_length=protein_args.force_exact_length,
        sid_to_target_info=SID_TO_TARGET_INFO,
        protein_action_space=PROTEIN_ACTION_SPACE,
        policy_provenance=_policy_provenance,
    )

    logger.info("ProteinGRPOTrainer initialized")
    logger.info(f"  Trainable parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,}")
    logger.info(f"  Total parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")

    # Optimization
    logger.info("=" * 60)
    logger.info("Starting GRPO training...")
    logger.info("=" * 60)

    resume_ckpt = getattr(grpo_config, "resume_from_checkpoint", None)
    if resume_ckpt:
        resume_path = Path(resume_ckpt).resolve()
        if not resume_path.is_dir():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        current_checkpoint_dir = Path(grpo_config.output_dir).resolve()
        allow_incompatible_resume = "allow_incompatible_resume" in str(
            current_checkpoint_dir
        )
        checkpoint_name_ok = re.fullmatch(r"checkpoint-[0-9]+", resume_path.name)
        same_clean_run = resume_path.parent == current_checkpoint_dir
        if (
            not checkpoint_name_ok or not same_clean_run
        ) and not allow_incompatible_resume:
            raise RuntimeError(
                "Refusing to resume a checkpoint outside the current run directory. "
                "Use an output name containing 'allow_incompatible_resume' to "
                "explicitly permit an external checkpoint."
            )
        logger.info(f"Resuming training from checkpoint: {resume_path}")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    logger.info("Training complete")

    # Final checkpoint
    final_save_dir = os.path.join(grpo_config.output_dir, "final_model")
    trainer.save_model(final_save_dir)
    TOKENIZER.save_pretrained(final_save_dir)
    logger.info(f"Model saved to {final_save_dir}")

    # Post-training validation
    if protein_args.validate_after_training and val_list:
        logger.info("Running post-training validation...")
        metrics = evaluate_all_metrics(
            model=trainer.model,
            tokenizer=TOKENIZER,
            val_list=val_list,
            pdb_val_dir=protein_args.pdb_val_dir,
            device=DEVICE,
            num_samples=protein_args.val_num_generations,
            max_structures=len(val_list),
            temperature=protein_args.eval_temperature,
            top_p=protein_args.eval_top_p,
            force_exact_length=protein_args.force_exact_length,
            output_dir=log_dir,  # Diagnostic output root.
            step=-1,  # -1 indicates post-training
            reward_weights=val_reward_weights,
            protein_action_space=PROTEIN_ACTION_SPACE,
        )

        if trainer.is_world_process_zero():
            val_results_path = os.path.join(log_dir, "val_results.pkl")
            with open(val_results_path, "wb") as f:
                pickle.dump(metrics, f)
            logger.info(f"Validation results saved to {val_results_path}")

    logger.info("Training workflow complete")



if __name__ == "__main__":
    main()
