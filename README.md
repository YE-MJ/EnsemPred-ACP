# EnsemblePred-ACP

Project description will be added later.


## Installation Guide

This code requires Python 3. For simplicity, I recommend using Anaconda. If conda is not installed on your system, you can download it [here](https://docs.anaconda.com/miniconda/).

1. (Optional) Create a conda environment for this workshop and activate it:

```bash
conda create -n EnsemPred-ACP python=3.10.15
source activate EnsemPred-ACP
```

2. Install the environment:

```bash
conda env update --file environment.yaml
```


3. (OPtional) If the pip packages are not installed correctly or there are conflicts:

```bash
pip install -r requirements.txt
```

- You may need to check for NVIDIA-related packages or PyTorch and CUDA compatibility issues in advance.
Refer to the PyTorch installation guide for [details](https://pytorch.org/get-started/locally/) on setting up PyTorch with various CUDA versions.

## Dataset

antiCP2.0: Download Standalone Version of [AntiCP2.0](https://webs.iiitd.edu.in/raghava/anticp2/download.php)

## Usage

How to run the test set using the pre-trained weights:

```bash
python test.py
```

How to run the independent set using the pre-trained weights:

```bash
python independent.py
```

How to run a custom dataset in TXT format with FASTA structure.
- Place the dataset in the 'data folder' and modify the 'filepath' at 'line 254' in 'test.py' or 'independent.py' before running the script.

If you want to run a simple sequence in FASTA format.
- Use this [website](http://thegleelab.org/EnsemPred-ACP/) 여기 웹사이트를 이용

## Author

To be added once the paper is published.

## Reference

To be added once the paper is published.