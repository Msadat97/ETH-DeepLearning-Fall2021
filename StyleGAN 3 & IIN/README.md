# Codes for reproducing StyleGAN3 related results

This part can be used to reproduce our results that are based on the Stylegan3 model.

## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.1 or later.


## StyleFlow data

Please download the StyleFlow data from https://github.com/RameenAbdal/StyleFlow/tree/master/data

## Computing DCI

Run the following code to get the DCI results for stylegan3

```[shell]
#please set corresponding directories at the beginning of the file

python get_latent_codes.py
 python convert_grads.py
cd StyleSpace
bash annotate.sh
bash dci-train.sh
bash dci-test.sh
```
## Face Segmentation for the generated images
Please follow the instructions here to annotate the images: [https://github.com/betterze/StyleSpace]

## Manipulations using localized and attribute channels
First, you need to compute the overlap score for the gradients. This can be done via 
```
cd StyleSpace
bash ./bash_script/align_mask.sh
```
Manipulations can be done via `manipulate.py`. For example, by running
```
python manipulate.py
```

## Training INN network
For training the INN network, you can set the configs inside `inn/config_factor.yaml` and run

```
python train_iin.py 
```

## testing INN network
For training the INN network, you can set the configs inside `inn/config_factor.yaml` and run
```
python test_iin.py 
```