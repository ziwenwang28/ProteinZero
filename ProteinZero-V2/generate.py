#!/usr/bin/env python3
"""Generate protein sequences conditioned on backbone structures.

These checkpoints are intended for structures up to 100 residues.
Behavior outside this length range has not been characterized.

The script loads a ProteinMPNN checkpoint, designs all chains found in each
input PDB file, and writes the generated sequences to a tab-separated file.
For multichain structures, chain sequences are separated by ``/`` in the
``sequence`` column.

No structure prediction, scoring, or evaluation is performed.

Dependencies:
    - Python 3.10 or later
    - NumPy
    - PyTorch
    - The bundled ``ProteinMPNN/protein_mpnn_utils.py`` implementation

Output columns:
    structure_id
        Input PDB filename without the ``.pdb`` suffix.
    sample_id
        One-based sample index for the corresponding backbone.
    sequence
        Generated amino acid sequence. Multiple chains are separated by ``/``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


LOGGER = logging.getLogger("proteinzero.generate")

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
PROTEINMPNN_DIRECTORY = REPOSITORY_ROOT / "ProteinMPNN"
PROTEINMPNN_UTILS = PROTEINMPNN_DIRECTORY / "protein_mpnn_utils.py"

if not PROTEINMPNN_UTILS.is_file():
    raise ImportError(
        "The bundled ProteinMPNN implementation was not found at "
        f"{PROTEINMPNN_UTILS}."
    )

sys.path.insert(0, str(PROTEINMPNN_DIRECTORY))

from protein_mpnn_utils import (  # noqa: E402
    ProteinMPNN,
    _S_to_seq,
    parse_PDB,
    tied_featurize,
)


MODEL_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
STANDARD_AMINO_ACIDS = frozenset(MODEL_ALPHABET[:-1])
MAX_CHARACTERIZED_LENGTH = 100

OMIT_UNKNOWN_AMINO_ACID = np.asarray(
    [amino_acid == "X" for amino_acid in MODEL_ALPHABET],
    dtype=np.float32,
)
ZERO_AMINO_ACID_BIAS = np.zeros(len(MODEL_ALPHABET), dtype=np.float32)


@dataclass(frozen=True)
class BackboneFeatures:
    """ProteinMPNN input features for one backbone."""

    coordinates: torch.Tensor
    native_sequence: torch.Tensor
    coordinate_mask: torch.Tensor
    chain_mask: torch.Tensor
    chain_encoding: torch.Tensor
    residue_indices: torch.Tensor
    mutable_position_mask: torch.Tensor
    omitted_amino_acid_mask: torch.Tensor
    pssm_coefficients: torch.Tensor
    pssm_bias: torch.Tensor
    pssm_log_odds: torch.Tensor
    residue_bias: torch.Tensor
    chain_lengths: tuple[int, ...]
    total_length: int


def positive_integer(value: str) -> int:
    """Parse a strictly positive integer."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an integer, received {value!r}."
        ) from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("The value must be greater than zero.")
    return parsed


def nonnegative_seed(value: str) -> int:
    """Parse a seed accepted by PyTorch."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an integer seed, received {value!r}."
        ) from exc

    if not 0 <= parsed <= (2**63 - 1):
        raise argparse.ArgumentTypeError(
            "The seed must be between 0 and 2^63 - 1."
        )
    return parsed


def positive_finite_float(value: str) -> float:
    """Parse a finite floating-point value greater than zero."""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a floating-point value, received {value!r}."
        ) from exc

    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "The value must be finite and greater than zero."
        )
    return parsed


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate ProteinMPNN sequences conditioned on backbone PDB files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to a ProteinZero V2 ProteinMPNN checkpoint containing a "
            "plain model state dictionary."
        ),
    )
    parser.add_argument(
        "--pdb_dir",
        type=Path,
        required=True,
        help=(
            "Directory containing input PDB files. Files are discovered "
            "non-recursively and processed in filename order."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the tab-separated output file.",
    )
    parser.add_argument(
        "--num_seqs",
        type=positive_integer,
        default=8,
        help="Number of sequences to generate for each backbone.",
    )
    parser.add_argument(
        "--temperature",
        type=positive_finite_float,
        default=0.1,
        help=(
            "Sampling temperature. ProteinMPNN suggests values spanning "
            "0.1 to 0.3. Default: 0.1."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Execution device: auto, cpu, cuda, or cuda:N. The auto setting "
            "uses CUDA when available and otherwise uses the CPU."
        ),
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_seed,
        default=0,
        help=(
            "Random seed for sequence generation. Reproducibility is "
            "expected within the same software and hardware environment."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def resolve_device(specification: str) -> torch.device:
    """Resolve and validate the requested execution device."""

    normalized = specification.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Invalid device specification: {specification!r}."
        ) from exc

    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported.")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA device was requested, but CUDA is not available."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"{torch.cuda.device_count()} device(s) were detected."
            )

    return device


def discover_pdb_files(pdb_directory: Path) -> list[Path]:
    """Return a validated, deterministic list of input PDB files."""

    if not pdb_directory.is_dir():
        raise FileNotFoundError(
            f"The PDB directory does not exist: {pdb_directory}"
        )

    pdb_files = sorted(
        (
            path
            for path in pdb_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdb"
        ),
        key=lambda path: path.name.casefold(),
    )
    if not pdb_files:
        raise ValueError(
            f"No PDB files were found in {pdb_directory}."
        )

    identifiers: set[str] = set()
    for pdb_path in pdb_files:
        identifier = pdb_path.stem
        if not identifier or any(
            ord(character) < 32 or ord(character) == 127
            for character in identifier
        ):
            raise ValueError(
                f"Invalid structure identifier derived from {pdb_path.name!r}."
            )

        normalized_identifier = identifier.casefold()
        if normalized_identifier in identifiers:
            raise ValueError(
                "Input PDB filenames must have unique case-insensitive stems; "
                f"duplicate identifier: {identifier!r}."
            )
        identifiers.add(normalized_identifier)

    return pdb_files


def load_model(checkpoint: Path, device: torch.device) -> ProteinMPNN:
    """Instantiate the checkpoint-compatible architecture and load its weights."""

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"The checkpoint does not exist: {checkpoint}"
        )

    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise TypeError(
            "The checkpoint must contain a non-empty model state dictionary."
        )
    if any(not isinstance(key, str) for key in state_dict):
        raise TypeError("Every checkpoint key must be a string.")
    if any(not isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise TypeError(
            "The checkpoint must contain tensors only; training-state objects "
            "are not accepted."
        )

    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        dropout=0.0,
        k_neighbors=48,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return model


def seed_generators(seed: int) -> None:
    """Seed all random generators used by the sampling implementation."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def featurize_backbone(
    pdb_path: Path,
    device: torch.device,
) -> BackboneFeatures:
    """Parse and featurize one complete backbone."""

    parsed_structures = parse_PDB(str(pdb_path), ca_only=False)
    if len(parsed_structures) != 1:
        raise ValueError(
            f"Expected one parsed structure in {pdb_path.name}, received "
            f"{len(parsed_structures)}."
        )

    parsed_structure = parsed_structures[0]
    if int(parsed_structure.get("num_of_chains", 0)) <= 0:
        raise ValueError(f"No protein chains were found in {pdb_path.name}.")

    (
        _original_sequence,
        coordinates,
        native_sequence,
        coordinate_mask,
        lengths,
        chain_mask,
        chain_encoding,
        _chain_lists,
        _visible_chain_lists,
        _masked_chain_lists,
        masked_chain_length_lists,
        mutable_position_mask,
        omitted_amino_acid_mask,
        residue_indices,
        _dihedral_mask,
        _tied_position_lists,
        pssm_coefficients,
        pssm_bias,
        pssm_log_odds,
        residue_bias,
        _tied_beta,
    ) = tied_featurize(
        [parsed_structure],
        device,
        chain_dict=None,
        fixed_position_dict=None,
        omit_AA_dict=None,
        tied_positions_dict=None,
        pssm_dict=None,
        bias_by_res_dict=None,
        ca_only=False,
    )

    total_length = int(lengths[0])
    if total_length <= 0:
        raise ValueError(f"The structure is empty: {pdb_path.name}.")
    if total_length > MAX_CHARACTERIZED_LENGTH:
        raise ValueError(
            f"{pdb_path.name} contains {total_length} residues; these "
            f"checkpoints are intended for structures containing at most "
            f"{MAX_CHARACTERIZED_LENGTH} residues."
        )

    residues_with_complete_backbone = int(
        coordinate_mask[0, :total_length].sum().item()
    )
    if residues_with_complete_backbone != total_length:
        raise ValueError(
            f"{pdb_path.name} has complete N, CA, C, and O coordinates for "
            f"{residues_with_complete_backbone} of {total_length} residues. "
            "Generation requires a complete backbone at every position."
        )

    chain_lengths = tuple(
        int(length) for length in masked_chain_length_lists[0]
    )
    if not chain_lengths or sum(chain_lengths) != total_length:
        raise RuntimeError(
            f"ProteinMPNN returned inconsistent chain lengths for "
            f"{pdb_path.name}."
        )

    return BackboneFeatures(
        coordinates=coordinates,
        native_sequence=native_sequence,
        coordinate_mask=coordinate_mask,
        chain_mask=chain_mask,
        chain_encoding=chain_encoding,
        residue_indices=residue_indices,
        mutable_position_mask=mutable_position_mask,
        omitted_amino_acid_mask=omitted_amino_acid_mask,
        pssm_coefficients=pssm_coefficients,
        pssm_bias=pssm_bias,
        pssm_log_odds=pssm_log_odds,
        residue_bias=residue_bias,
        chain_lengths=chain_lengths,
        total_length=total_length,
    )


def restore_chain_boundaries(
    sequence: str,
    chain_lengths: tuple[int, ...],
) -> str:
    """Insert separators between chains in a generated sequence."""

    if len(sequence) != sum(chain_lengths):
        raise RuntimeError(
            "The generated sequence length does not match the parsed backbone."
        )

    chain_sequences: list[str] = []
    offset = 0
    for chain_length in chain_lengths:
        next_offset = offset + chain_length
        chain_sequences.append(sequence[offset:next_offset])
        offset = next_offset
    return "/".join(chain_sequences)


def sample_sequence(
    model: ProteinMPNN,
    features: BackboneFeatures,
    temperature: float,
    device: torch.device,
) -> str:
    """Generate one sequence for a featurized backbone."""

    decoding_noise = torch.randn(
        features.chain_mask.shape,
        device=device,
    )
    pssm_log_odds_mask = (features.pssm_log_odds > 0.0).float()

    with torch.inference_mode():
        sampled = model.sample(
            features.coordinates,
            decoding_noise,
            features.native_sequence,
            features.chain_mask,
            features.chain_encoding,
            features.residue_indices,
            mask=features.coordinate_mask,
            temperature=temperature,
            omit_AAs_np=OMIT_UNKNOWN_AMINO_ACID,
            bias_AAs_np=ZERO_AMINO_ACID_BIAS,
            chain_M_pos=features.mutable_position_mask,
            omit_AA_mask=features.omitted_amino_acid_mask,
            pssm_coef=features.pssm_coefficients,
            pssm_bias=features.pssm_bias,
            pssm_multi=0.0,
            pssm_log_odds_flag=False,
            pssm_log_odds_mask=pssm_log_odds_mask,
            pssm_bias_flag=False,
            bias_by_res=features.residue_bias,
        )

    sequence = _S_to_seq(sampled["S"][0], features.chain_mask[0])
    if len(sequence) != features.total_length:
        raise RuntimeError(
            "ProteinMPNN returned a sequence whose length does not match the "
            "input backbone."
        )

    unexpected_symbols = sorted(set(sequence) - STANDARD_AMINO_ACIDS)
    if unexpected_symbols:
        raise RuntimeError(
            "ProteinMPNN returned non-standard amino-acid symbols: "
            + ", ".join(unexpected_symbols)
        )

    return restore_chain_boundaries(sequence, features.chain_lengths)


def validate_output_path(
    output: Path,
    checkpoint: Path,
    pdb_files: list[Path],
    overwrite: bool,
) -> None:
    """Validate the destination before generation begins."""

    if output.exists() and output.is_dir():
        raise IsADirectoryError(
            f"The output path is a directory: {output}"
        )
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"The output file already exists: {output}. "
            "Pass --overwrite to replace it."
        )

    resolved_output = output.resolve()
    protected_paths = [checkpoint.resolve()]
    protected_paths.extend(path.resolve() for path in pdb_files)
    if resolved_output in protected_paths:
        raise ValueError(
            "The output path must not refer to the checkpoint or an input "
            "PDB file."
        )


def generate_sequences(args: argparse.Namespace) -> None:
    """Run sequence generation and atomically write the output file."""

    checkpoint = args.checkpoint.expanduser()
    pdb_directory = args.pdb_dir.expanduser()
    output = args.output.expanduser()
    device = resolve_device(args.device)
    pdb_files = discover_pdb_files(pdb_directory)
    validate_output_path(
        output,
        checkpoint,
        pdb_files,
        args.overwrite,
    )

    model = load_model(checkpoint, device)

    # Seed after model construction so the seed controls sampling only.
    seed_generators(args.seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    sequence_count = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.writer(
                handle,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writerow(["structure_id", "sample_id", "sequence"])

            for pdb_path in pdb_files:
                features = featurize_backbone(pdb_path, device)
                for sample_id in range(1, args.num_seqs + 1):
                    sequence = sample_sequence(
                        model,
                        features,
                        args.temperature,
                        device,
                    )
                    writer.writerow([pdb_path.stem, sample_id, sequence])
                    sequence_count += 1

                LOGGER.info(
                    "Generated %d sequence(s) for %s.",
                    args.num_seqs,
                    pdb_path.stem,
                )

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    LOGGER.info(
        "Wrote %d sequence(s) for %d backbone(s).",
        sequence_count,
        len(pdb_files),
    )


def main() -> None:
    """Command-line entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = parse_arguments()
    try:
        generate_sequences(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
