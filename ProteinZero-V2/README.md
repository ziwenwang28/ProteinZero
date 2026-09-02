# ProteinZero V2 Teaser Checkpoints

This directory contains the ProteinZero V2 teaser checkpoints for
structure-conditioned protein sequence design. The checkpoints use
ProteinMPNN as the base model.

ProteinZero V1 was built on InstructPLM, whose architecture constrains the
sequence length it can process. ProteinZero V2 uses ProteinMPNN as the base
model, which removes that constraint.

The checkpoints were trained with PARPO, a general-purpose reinforcement
learning algorithm for multi-objective and multi-turn optimization. Its
applicability is not limited to protein design. A complete formulation and
derivation will be presented in the forthcoming ProteinZero V2 paper.

Reward signals combine structural feedback from ESMFold with the fast ΔΔG
score introduced in ProteinZero V1.

These checkpoints are intended for structures of up to 100 residues. Behavior
beyond this range has not been characterized.

## Checkpoints

| File                                     | Structural feedback |
| ---------------------------------------- | ------------------- |
| `checkpoints/proteinzero_v2_esmfold1.pt` | ESMFold1            |
| `checkpoints/proteinzero_v2_esmfold2.pt` | ESMFold2            |

The two checkpoints were trained with different structural feedback models,
as listed above.

The checkpoints are plain ProteinMPNN state dictionaries. Instantiate
`ProteinMPNN` with `ca_only=False, num_letters=21, node_features=128,
edge_features=128, hidden_dim=128, num_encoder_layers=3,
num_decoder_layers=3, augment_eps=0.0, dropout=0.0, k_neighbors=48` and load
with `strict=True`. Weights only; no optimizer or training state.

## Sequence generation

`generate.py` locates repository components relative to its own file and can
be invoked from any working directory. Replace the paths below with those of
your own installation.

```bash
python /path/to/ProteinZero/ProteinZero-V2/generate.py \
  --checkpoint /path/to/ProteinZero/ProteinZero-V2/checkpoints/proteinzero_v2_esmfold1.pt \
  --pdb_dir /path/to/backbones \
  --output /path/to/generated_sequences.tsv \
  --num_seqs 8 \
  --temperature 0.1 \
  --device auto \
  --seed 0 \
  --overwrite
```

### Arguments

| Argument        | Description                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------- |
| `--checkpoint`  | Path to a ProteinZero V2 checkpoint. Required.                                                    |
| `--pdb_dir`     | Directory containing input PDB files. Files are read non-recursively in filename order. Required. |
| `--output`      | Destination path for the tab-separated output file. Required.                                     |
| `--num_seqs`    | Number of sequences generated per backbone. Default: `8`.                                         |
| `--temperature` | ProteinMPNN sampling temperature. Suggested values span `0.1`–`0.3`. Default: `0.1`.              |
| `--device`      | Execution device: `auto`, `cpu`, `cuda`, or `cuda:N`. Default: `auto`.                            |
| `--seed`        | Random seed for sequence generation. Default: `0`.                                                |
| `--overwrite`   | Replace the output file if it already exists.                                                     |

The output is a tab-separated file with three columns:

```text
structure_id	sample_id	sequence
example_backbone	1	MSHWWLVGFKGWNLVTLLALAILVAIIGLLVVILALIAAIIGVP
```

`structure_id` is the input PDB filename without its suffix, and `sample_id`
is a one-based index. For multichain structures, chain sequences are separated
by `/`.

The script generates sequences only. Folding, scoring, and evaluation are left
to the user.

## Dependencies

`generate.py` requires Python 3.10 or later, PyTorch, NumPy, and the
ProteinMPNN implementation included in this repository.

```bash
python -m pip install torch numpy
```

ESMFold, OpenFold, fair-esm, and TM-align are not required for sequence
generation. They are used only by the ProteinZero V1 training and scoring
workflow.

## Planned releases

> **Full checkpoints** with detailed analysis will be released in a forthcoming
> update. An **interactive interface** for protein sequence design is also
> planned.

## Citation

If you use these checkpoints, please cite ProteinZero together with the
methods on which this work builds.

```bibtex
@article{wang2025proteinzero,
  title={Proteinzero: Self-improving protein generation via online reinforcement learning},
  author={Wang, Ziwen and Fan, Jiajun and Guo, Ruihan and Nguyen, Thao and Ji, Heng and Liu, Ge},
  journal={arXiv preprint arXiv:2506.07459},
  year={2025}
}

@article{dauparas2022robust,
  title={Robust deep learning-based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022},
  doi={10.1126/science.add2187},
  publisher={American Association for the Advancement of Science}
}

@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yaniv and others},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023},
  doi={10.1126/science.ade2574},
  publisher={American Association for the Advancement of Science}
}

@misc{candido2026language,
  title={Language Modeling Materializes a World Model of Protein Biology},
  author={Candido, Salvatore and Hayes, Thomas and Derry, Alexander and Rao, Roshan and Lin, Zeming and Verkuil, Robert and others},
  year={2026},
  url={https://www.biorxiv.org/content/10.64898/2026.06.03.729735},
  note={Preprint}
}
```
