# EchoNet-Dynamic-Reproduction

## Model Weights

The `best.pt` checkpoint in `/output/video/` is trained on the [EchoNet-Dynamic](https://echonet.github.io/dynamic/) dataset.  
This file is shared **for research and educational purposes only**.  
Please cite the original dataset and publication if you use this model:

> [Video-based AI for beat-to-beat assessment of cardiac function](https://www.nature.com/articles/s41586-020-2145-8)
>
> David Ouyang, Bryan He, Amirata Ghorbani, Neal Yuan, Joseph Ebinger, Curt P. Langlotz, Paul A. Heidenreich,
> Robert A. Harrington, David H. Liang, Euan A. Ashley, and James Y. Zou. Nature, March 25, 2020.
> *Nature* 580, 252–256 (2020). https://doi.org/10.1038/s41586-020-2145-8


This repository reproduces the baseline model training and evaluation
of the [EchoNet-Dynamic](https://echonet.github.io/dynamic/) dataset for
left ventricular ejection fraction (LVEF) estimation using PyTorch.

It includes:
- Baseline segmentation and regression model runs
- Performance metrics (MAE, RMSE, R²)
- Visualization results (scatter & ROC curves)


## Pretrained Weights
Pretrained model weights (`best.pt`) are hosted on Hugging Face:

[Download from Hugging Face](https://huggingface.co/janalexei98/echonet-dynamic-best-pt/tree/main)
