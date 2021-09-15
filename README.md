# ViCatDA
Code release for `Vicinal and categorical domain adaptation`, which has been accepted by Pattern Recognition. 

The paper is available [here](https://www.sciencedirect.com/science/article/pii/S0031320321000947) or at the [arXiv archive](https://arxiv.org/abs/2103.03460).

## Requirements
- python 3.6.4
- pytorch 1.4.0
- torchvision 0.5.0

## Data preparation
The structure of the used datasets is shown in the folder `./data/datasets/`. 

The original datasets can be downloaded [here](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md).

## Model training
1. Replace paths and domains in run.sh with those in one's own system. 
2. Install necessary python packages.
3. Run command `sh run.sh`.

The results are saved in the folder `./checkpoints/`.

## Article citation
```
@article{vicatda,
author = {Hui Tang and Kui Jia},
title = {Vicinal and categorical domain adaptation},
journal = {Pattern Recognition},
year = {2021},
volume = {115},
pages = {107907},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2021.107907},
url = {https://www.sciencedirect.com/science/article/pii/S0031320321000947},
}
```
