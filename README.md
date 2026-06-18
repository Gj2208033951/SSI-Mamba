
# SSI-Mamba

SSI-Mamba is a multimodal Siamese deep learning framework for silencer–silencer interaction (SSI) prediction.

## Environment

The software environment used in this study is provided through:

* requirements.txt

Main dependencies:

* Python 3.10
* PyTorch 2.0.0
* CUDA 11.8
* Mamba-SSM 1.0.1
* Captum 0.9.0
* SHAP 0.49.1

## Input Data

The model takes two modalities as input:

### DNA Sequence

Each anchor is represented as a 200 × 6 matrix:

* A
* C
* G
* T
* N
* Anchor Mask

### Histone Modification Signals

Each anchor is represented as a 200 × 6 matrix:

* H3K4me3
* H3K9me3
* H3K36me3
* H3K4me1
* H3K9ac
* H3K27ac

The final model therefore receives:

* Anchor1 Sequence
* Anchor2 Sequence
* Anchor1 Histone Signal
* Anchor2 Histone Signal

## Data Sources

### Reference Genome

* hg38 (human)
* mm10 (mouse)

### Histone Modification Signals

Histone modification bigWig files can be downloaded from ENCODE:

https://www.encodeproject.org/

The accession identifiers used in this study are listed in Supplementary Table S1.

### Chromatin Interaction Data

Chromatin interaction datasets were obtained from Loop Catalog.

The GEO accession identifiers used in this study are listed in Supplementary Table S2.

## Training

Modify the paths in Train.py:

* reference genome (.fa)
* histone bigWig files
* DNA dataset

Run:

```bash
python Train.py
```

## Evaluation

Download the trained model weights and modify the checkpoint path in test.py.

Run:

```bash
python test.py
```

## Model Weights

You can also directly make predictions by using the trained weights.

## Reproducibility

All experiments were conducted using:

* NVIDIA A800-SXM4-80GB
* PyTorch 2.0.0
* CUDA 11.8
* Mamba-SSM 1.0.1

