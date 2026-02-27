# Stochastic Self-Organization in Multi-Agent Systems [ICLR 2026]

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)](https://openreview.net/forum?id=rS3Jb9AAej)
[![OpenReview](https://img.shields.io/badge/OpenReview-Forum-f31f1f)](https://openreview.net/forum?id=rS3Jb9AAej)
[![arXiv](https://img.shields.io/badge/arXiv-2510.00685-teal)](https://arxiv.org/abs/2510.00685)


> [**Stochastic Self-Organization in Multi-Agent Systems [ICLR 2026]**](https://openreview.net/forum?id=rS3Jb9AAej)<br>
> [Nurbek Tastan](https://tnurbek.github.io/), [Samuel Horvath](https://sites.google.com/view/samuelhorvath/home), [Karthik Nandakumar](https://www.cse.msu.edu/~nandakum/)<br>
> The Fourteenth International Conference on Learning Representations (ICLR), 2026<br> 


## Abstract 

Large Language Models (LLMs) have enabled multi-agent systems (MAS) where agents collaborate to solve tasks beyond the reach of a single model. Yet most existing approaches rely on fixed topologies, pretrained graph generators, optimization over edges, or external LLM judges, thereby adding complexity. We introduce a response-conditioned framework that adapts communication on the fly. Agents independently generate answers and assess peer contributions using a Shapley~value-inspired approximation. A directed acyclic graph (DAG) is then constructed to route information from high-contribution agents to others, ensuring stable and efficient message passing without the need for additional supervision or training. We provide a theoretical analysis showing that multiple agents increase the chance of correctness and that the correct answers naturally dominate information flow. Experiments with both strong and weak LLM backends demonstrate robust performance, with significant gains in the weak regime where prior methods collapse.




## Code 

To appear soon. 


## 📖 Citation 
If you like our work, please consider citing us: 

```bibtex
@inproceedings{tastan2026stochastic,
    title={Stochastic Self-Organization in Multi-Agent Systems},
    author={Nurbek Tastan and Samuel Horv{\'a}th and Karthik Nandakumar},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=rS3Jb9AAej}
}
```