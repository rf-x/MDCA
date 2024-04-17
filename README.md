# MDCA: Multimodal dual cause analysis

This repo provides the source code and data for the paper Dual Causes Generation Assisted Model for Multimodal Aspect-based Sentiment Classification

----------

## Overview

<p align="center">
  <img src="./framework.png" width="800"/>
</p>

----------

## Code Usage

----------

### Requirements

- Python == 3.11.14
- PyTorch == 2.0.1
- transformers == 4.32.1

----------

### Dataset
1. Text data: Twitter2015 ([twitter2015](data/twitter2015)) and Twitter2017 ([twitter2017](data/twitter2017)).
2. Image features data: Download from [Google Drive](https://drive.google.com/drive/folders/1F-cjyNCm57gpJd0Rt4zWJCjt_mvbOWwu?usp=sharing). Put them to the fold `data/twitter2015` and `data/twitter2017`.

----------

### Pretrained Models
- Use the BART as the backbone: Download [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) to `pretrained/bart-large-cnn`.
- Use the Flan-T5 as the backbone: Download [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) to `pretrained/flan-t5-base`.

###  Usage
1. Download the file of image features from [Google Drive](https://drive.google.com/drive/folders/1F-cjyNCm57gpJd0Rt4zWJCjt_mvbOWwu?usp=sharing). And put it to the fold `data/twitter2015` and `data/twitter2017`
2. Download the pre-trained `bart-large-cnn` and `flan-t5-base` from [transformers library](https://huggingface.co/models). And put them to the folder `pretrained`.
3. Train and evaluate the model:

```
# BART
python run_bart.py
# Flan-T5
python run_flant5.py
```