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

[Dataset](https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a)

It includes:
- Baseline segmentation and regression model runs
- Performance metrics (MAE, RMSE, R²)
- Visualization results (scatter & ROC curves)

## Pretrained Weights
Pretrained model weights (`best.pt`) are hosted on Hugging Face:

[Download from Hugging Face](https://huggingface.co/janalexei98/echonet-dynamic-best-pt/tree/main)


## How to use

1. please clone original [repo](https://github.com/echonet/dynamic)

2. install
```
pip install -e .
pip install torch torchvision scikit-learn scikit-image tqdm opencv-python
```

3. train it
```
# Train segmentation
echonet segmentation --data_dir [path_to_dataset] --num_epochs 1

# Train EF regression (video model)
echonet video --data_dir [path_to_dataset] --num_epochs 45

# Run inference
echonet video --data_dir [path_to_dataset] --run_test --num_epochs 0
```

## Original output Visualization

1. test_roc
<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/a932c553-fb0c-45fc-8101-f4b7d55f6921" />

2. test_scatter
<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/ec3460b6-476a-4947-9a81-42e1fadfff2e" />

3. val_roc
<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/45cb5134-9ea5-4a82-9915-57a833d20bf8" />

4. val_scatter
<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/89823f86-0235-4e7e-aee2-f1acc3427cc6" />

# EchoNet-Dynamic Baseline (R(2+1)D-18) — Results Summary

Recomputed locally from:
- `val_predictions.csv`, `test_predictions.csv` (per-clip predictions)
- Averaged to per-video and merged with `FileList.csv`.

## Metrics

| Split | Level     | MAE | RMSE | R² |
|:-----:|:---------:|----:|-----:|---:|
| VAL   | per-clip  | 4.10 | 5.42 | 0.79 |
| VAL   | per-video | 3.90 | 5.10 | 0.83 |
| TEST  | per-clip  | 4.20 | 5.53 | 0.78 |
| TEST  | per-video | 3.98 | 5.25 | 0.82 |

> *Per-video = average predictions over multiple clips of the same video before computing metrics.*


## Citation
This implementation is based on:
Ouyang D, et al. *Video-based AI for beat-to-beat assessment of cardiac function*. Nature, 2020.  
[https://echonet.github.io/dynamic/](https://echonet.github.io/dynamic/)


