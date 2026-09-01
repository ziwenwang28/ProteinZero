# ProteinZero

### Self-Improving Protein Generation via Online Reinforcement Learning

Welcome to the official GitHub repository for ProteinZero.

## ProteinZero

ProteinZero is an online reinforcement learning framework that enables protein
generative models to continuously improve by learning from their own outputs.

Instead of optimizing a single proxy, ProteinZero brings together multiple
objectives that matter in protein design, including designability, predicted
thermal stability, and sequence diversity. Its feedback pipeline combines
structural evaluation, a fast self-derived ΔΔG predictor, and protein-embedding
diversity regularization to support scalable iterative optimization.

![ProteinZero framework](./assets/main_chart.jpg)

## Repository contents

| Path                          | Contents                                                         |
| ----------------------------- | ---------------------------------------------------------------- |
| `train_proteinzero_grpo.py`   | GRPO training for structure-conditioned sequence design          |
| `run_proteinzero_grpo.sh`     | Training launcher                                                |
| `preprocess.py`               | Generates structure embeddings from backbone PDB files           |
| `INSTALL.md`                  | Installation, data preparation, and usage                        |
| `patches/`                    | Source patches required to build OpenFold and fair-esm           |
| `MPNN-ProGen2-xlarge-CATH42/` | InstructPLM model code                                           |
| `ProteinMPNN/`                | ProteinMPNN implementation and weights                           |
| `ProteinZero-V2/`             | ProteinZero V2 teaser checkpoints and sequence generation script |

Start with [INSTALL.md](./INSTALL.md) for environment setup, model and dataset
retrieval, and how to launch training.

The CATH 4.3 chain-level dataset used in this work is available at
[ZiwenWang28/cath43-chain-level](https://huggingface.co/datasets/ZiwenWang28/cath43-chain-level).

## Why Multi-Objective Reinforcement Learning Matters

Protein design is fundamentally multi-objective. A useful design should not only
fold into the intended structure, but ultimately be capable of satisfying
downstream functional and experimental requirements.

We believe that multi-objective RL will become increasingly important as protein
design expands from computational generation into a complete design–build–test–learn
pipeline. In that future, feedback may come directly from wet-lab experiments and
be returned to the RL system as learning signals, allowing protein generative
models to improve through real experimental outcomes rather than computational
proxies alone.

![The position of ProteinZero in a future protein-design pipeline](./assets/proteinzero_position.jpg)

ProteinZero provides a bridge between protein generative models and a future
closed-loop pipeline in which computational and experimental feedback can jointly
guide model self-improvement.

## ProteinZero V2: Pushing the Pareto Frontier

The strongest ProteinZero variant in our current study, ProteinZeroGRPO, builds
on Group Relative Policy Optimization (GRPO). Our exploration of multi-objective
protein design has since led us to a new reinforcement learning algorithm: PARPO.

Like PPO and GRPO, PARPO is intended to be a generalizable reinforcement learning
algorithm rather than a method specific to protein design. It is designed
specifically to target the exploration of Pareto frontiers in multi-objective
optimization, where progress requires navigating trade-offs among competing
objectives.

In ProteinZero V2, we apply PARPO to explore and push the Pareto frontier of
protein generation. ProteinZero V2 is a paper currently in the final stages of
preparation and will describe the motivation, formulation, and behavior of PARPO
in detail.

Teaser checkpoints trained with the ProteinZero V2 algorithm are available in
[ProteinZero-V2/](./ProteinZero-V2), together with a script for generating
sequences from backbone structures.

## Citation

```bibtex
@article{wang2025proteinzero,
  title={Proteinzero: Self-improving protein generation via online reinforcement learning},
  author={Wang, Ziwen and Fan, Jiajun and Guo, Ruihan and Nguyen, Thao and Ji, Heng and Liu, Ge},
  journal={arXiv preprint arXiv:2506.07459},
  year={2025}
}
```
