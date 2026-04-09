# Objective

> Binary classification of histopathology images cells as immune or non-immune
based on learned morphological features. We performed cross-validation across 
two imaging domains, hematoxylin & eosin (H&E) staining and peripheral blood 
smears (PBS), to investigate whether learned features are transferable.

# Datasets

1. [Immunocto](https://zenodo.org/records/11073373) (H&E)
2. [Acute Promyelocytic Leukemia (APL)](https://www.kaggle.com/datasets/eugeneshenderov/acute-promyelocytic-leukemia-apl) (PBS)

# Model

We used an ensemble model consisting of ResNet18 and a ViT Large-16 adopted from 
base architecture in `torchvision.models`. Training was done over $20$ epochs on
GPUs at Digital Research Alliance of Canada (DRAC). [UNI](https://github.com/mahmoodlab/UNI)
and [DinoBloom](https://github.com/marrlab/DinoBloom) were trained as baseline
models for state-of-the-art (SOTA) comparison. This directory contains code for
loading the two datasets and training all models discussed.

The final report is available [here](./final_report.pdf).
