
# StyleSpace Analysis for StyleFLOW model

This directory contains the code provided by the authors of StyleFlow along with the StyleSpace files required for performing StyleSpace analysis. We try to follow the StyleSpace analysis as closely as possible.

## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 64-bit Tensorflow 1.x 
* CUDA toolkit < 11


## Calculating DCI
To generate the W space, use the latent codes for the data present, which can be then manipulated between the channel 0-7 and 12-18 to serve as the sampled W space(num_samples,18,512).

The W latent vectors for StyleFlow can be downloaded at https://drive.google.com/file/d/1m2KzqbxCW_2AHSXiHEoBnf8Rh7M_nZX8/view?usp=sharing


Firstly switch to the StyleSpace directory: 
* `cd StyleSpace`

<!-- To generate the latent vectors in Z, W and S space along with the images, run
* `pythonGetCode.py [-h] [--save_path SAVE_PATH] [--num_samples NUM_SAMPLES] --mode SFlat [--resize RESIZE]` 

Using the generated latent vectors, the DCI metric can be calculated in the following steps

* `cd StyleSpace` -->
  
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

* `cd stylegan2-ada-pytorch`
* `python GetCode.py` 
* `python gradient_mask.py`

Follow the remaining steps in StyleSpace/bash_script/localized_channel.sh to generate the `semantic_top_32` file

To manipulate images, run:

* `python Manipulate.py`

## Citation

Rameen Abdal, Peihao Zhu, Niloy J. Mitra, and Peter
Wonka. Styleflow: Attribute-conditioned exploration
of stylegan-generated images using conditional continu-
ous normalizing flows. ACM Trans. Graph., 40(3), May
2021
