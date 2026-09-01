---
license: apache-2.0
pipeline_tag: text-generation
tags:
- biology
---

# InstrcutPLM
InstructPLM is a state-of-the-art protein design model based on [ProGen2](https://www.cell.com/cell-systems/abstract/S2405-4712(23)00272-7) 
and [ProteinMPNN](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9997061/) 
and trained on [CATH 4.2](https://www.cathdb.info/) dataset. 
It can design protein sequences that accurately conform to specified backbone structures.

<p align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/62a8397d839eeb3ef16a7566/1NRk65EImgBAFgvh8HJrA.png" alt="drawing" width="200"/>
</p>

Please visit our [repo](https://github.com/Eikor/InstructPLM) and [paper](https://biorxiv.org/cgi/content/short/2024.04.17.589642v1) for more information.

```
@article {Qiu2024.04.17.589642,
	author = {Jiezhong Qiu and Junde Xu and Jie Hu and Hanqun Cao and Liya Hou and Zijun Gao and Xinyi Zhou and Anni Li and Xiujuan Li and Bin Cui and Fei Yang and Shuang Peng and Ning Sun and Fangyu Wang and Aimin Pan and Jie Tang and Jieping Ye and Junyang Lin and Jin Tang and Xingxu Huang and Pheng Ann Heng and Guangyong Chen},
	title = {InstructPLM: Aligning Protein Language Models to Follow Protein Structure Instructions},
	elocation-id = {2024.04.17.589642},
	year = {2024},
	doi = {10.1101/2024.04.17.589642},
	publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2024/04/20/2024.04.17.589642},
	eprint = {https://www.biorxiv.org/content/early/2024/04/20/2024.04.17.589642.full.pdf},
	journal = {bioRxiv}
}
```

