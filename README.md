# Reference-Limited Compositional Zero-Shot Learning (RL-CZSL)

* **Title**: **[Reference-Limited Compositional Zero-Shot Learning](https://arxiv.org/pdf/2208.10046)**
* **Authors**: [Siteng Huang](https://kyonhuang.top/), [Qiyao Wei](https://qiyaowei.github.io/index.html), [Donglin Wang](https://milab.westlake.edu.cn/)
* **Institutes**: Zhejiang University, University of Cambridge, Westlake University
* **Conference**: Proceedings of the 2023 ACM International Conference on Multimedia Retrieval (ICMR 2023)
* **More details**: [[arXiv]](https://arxiv.org/pdf/2208.10046) | [[homepage]](https://kyonhuang.top/publication/reference-limited-CZSL)

**News**: We are in the process of cleaning up the source code and datasets. As I am currently away from university on weekdays, we hope to complete these tasks within a few weeks.

## Overview

* We introduce a new problem named **reference-limited compositional zero-shot learning (RL-CZSL)**, where given only a few samples of limited compositions, the model is required to generalize to recognize unseen compositions. This offers a more realistic and challenging environment for evaluating compositional learners.

<div align="middle"><img align="middle" style="max-width: 520px; width: 100%" src="https://kyonhuang.top/files/RLCZSL/RLCZSL-setting-comparison.png" /></div>

* We establish **two benchmark datasets with diverse compositional labels and well-designed data splits**, providing the required platform for systematically assessing progress on the task.

<div align="middle"><img align="middle" style="max-width: 540px; width: 100%" src="https://kyonhuang.top/files/RLCZSL/RLCZSL-dataset-stats.png" /></div>

* We propose a novel method, **Meta Compositional Graph Learner (MetaCGL)**, for the challenging RL-CZSL problem. Experimental results show that MetaCGL consistently outperforms popular baselines on recognizing unseen compositions. 

<div align="middle"><img align="middle" style="max-width: 540px; width: 100%" src="https://kyonhuang.top/files/RLCZSL/RLCZSL-main-results.png" /></div>

## Datasets

[Google Drive](https://drive.google.com/drive/folders/1zaNu4ay1ZiRstdFqk1S9hdwdi58YO1iF?usp=sharing)

## Citation

If you find this work useful in your research, please cite our paper:

```
@inproceedings{Huang2023RLCZSL,
    title={Reference-Limited Compositional Zero-Shot Learning},
    author={Siteng Huang and Qiyao Wei and Donglin Wang},
    booktitle = {Proceedings of the 2023 ACM International Conference on Multimedia Retrieval},
    month = {June},
    year = {2023}
}
```

## Acknowledgement

Our code references the following projects:

* [czsl](https://github.com/ExplainableML/czsl)
