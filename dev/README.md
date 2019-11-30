## Weakly Supervised Transfer Learning for Ventricle Segmentation v2


In this repo you will find notebooks and scripts which will allow you to find all the code necessary to replicate the experiments conducted in this project. This v2 version aims to be as simple as possible in terms of code readability and project structure. Improvements, suggestions and feedback are welcome.

Main library of the project is a custom forked version of [fastai v2] (https://github.com/KeremTurgutlu/fastai2/tree/extend_medical). Initially original code for experiments were written in a older version of PyTorch and Fast.ai. With this migration the code is now more organized and faster for mass consumption. Workflow adapted in this repo is pretty generic and potentially be used for other similar projects with different data or task with minimal changes.



### Setup

1) Follow the **Installation** steps in https://dev.fast.ai/ (including `fastai2.medical.imaging`). This installation step suggest creating a new environment, please do so. Let's say we created a new environment called `ventproject`.
2) Activate environment `conda activate ventproject`. Now we are in that environment.
3) Clone this project repo 
4) Clone the custom forked version of ![fastai v2] (https://github.com/KeremTurgutlu/fastai2/tree/extend_medical).
5) cd into custom fork and do `pip install -e .` this will install that repo as a pip package: `fastai2`



### Notebooks

In this project a technique called `literate programming` is used, meaning that most of the source code (80%) is generated using jupyter notebooks. This allows interactivity, fast develop and transparency for the users.

In this project you have the option to either use the notebooks or the scripts!

`0) scriptrunner.ipynb`: Implements utilities for running scripts in notebook env

`1a) dicom to tensor.ipynb`: Read DICOM data, resample, crop-pad and save. (has script)

`1b) skull strip.ipynb`: Skull strip data. (has script)

`1c) normalization.ipynb`: Normalize data to (0-1) scale for training. (has script)

`2) datasource.ipynb`: Defines fastai v2 `DataSource`

`3a) trainutils.ipynb`: Implements training utilities

`3b) traindenovo.ipynb`: End-to-end training for de novo mode (has script)

`3c) traintransfer.ipynb`: End-to-end training for weakly supervised transfer learning (has script)

### Scripts

Will be explained... 


### Configs

In this project you will need to have two `yaml` config files one for preparing data and one for defining transfer learned to pretrained model mapping. 

`data.yaml`: It is needed as we share raw DICOM images and this configuration will allow you to define where to read and where to write. It also will have csv metadata for each modality which has `train`, `valid`, `test1` and `test2` split information for each `StudyInstanceUID`.

`transfer_learning.yaml`: Here we define transfer learning model to pretrained model mappings for a given TASK - MODALITY combination. This is customizable depending on whic of the pretrained models you trained so far. By default all of our experiments left in this file.

















