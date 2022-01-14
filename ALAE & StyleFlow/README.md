
# StyleSpace Analysis for StyleALAE model

This directory contains the code provided by the authors of Adversarial Latent Autoencoders along with the StyleSpace files required for performing StyleSpace analysis for StyleALAE model. We try to follow the StyleSpace analysis as closely as possible.

## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 64-bit Python 3.8 and PyTorch 1.8.2 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.1 or later.
* Remaining libraries can be installed using `ALAE\requirements.txt`

## Pre-trained models

To download pre-trained StyleALAE models run:

* `cd ALAE`
* `python training_artifacts/download_all.py`

## Calculating DCI

Firstly switch to the ALAE directory: 
* `cd ALAE`

To generate the latent vectors in Z, W and S space along with the images, run
* `pythonGetCode.py [-h] [--save_path SAVE_PATH] [--num_samples NUM_SAMPLES] --mode SFlat [--resize RESIZE]` 

Using the generated latent vectors, the DCI metric can be calculated in the following steps

* `cd StyleSpace`
  
Download the classifiers 

* `mkdir metrics_checkpoint`
* `gdown https://drive.google.com/drive/folders/1MvYdWCBuMfnoYGptRH-AgKLbPTsIQLhl -O ./metrics_checkpoint --folder`

Annotate images

* `python GetAttribute.py -img_path  [img_path] -save_path [save_path] -classifer_path [classifer_path]`

Get and View DCI scores

* `python DCI.py -latent_path [latent_path]   -attribute_path [attribute_path] -save_path [save_path]`
* `python DCI.py -latent_path [latent_path]   -attribute_path [attribute_path] -save_path [save_path] -mode test`

## Finding Localized Channels 

Generate the gradients for each channel in StyleSpace S

* `cd ALAE`
* `python GetCode.py [-h] [--save_path SAVE_PATH] [--num_samples NUM_SAMPLES] --mode S [--resize RESIZE]` 
* `python gradient_mask.py [-h] [--Z_path Z_PATH] [--output_size OUTPUT_SIZE] [--sindex SINDEX] [--num_per NUM_PER]`

Follow the remaining steps in StyleSpace/bash_script/localized_channel.sh to generate the `semantic_top_32` file

To manipulate images, run:

* `python Manipulate.py [-h] [--base_path BASE_PATH] [--mode {Manipulate,ManipulateReal,AttributeDependent,PercentageLocalized}] [--Image_path IMAGE_PATH] [--W_file W_FILE] [--Semantic_file SEMANTIC_FILE] [--S_file S_FILE] [--Attribute_file ATTRIBUTE_FILE]`

## StyleFlow

To perform StyleSpace analysis on StyleFlow model, follow the steps in README_StyleFlow.md

## Citation
* Stanislav Pidhorskyi, Donald A. Adjeroh, and Gianfranco Doretto. Adversarial Latent Autoencoders. In *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020. 

* Wu, Zongze, Dani Lischinski, and Eli Shechtman. "Stylespace analysis: Disentangled controls for stylegan image generation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

