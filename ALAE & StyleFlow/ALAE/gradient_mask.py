from numpy import gradient
import torch
import torch.utils.data
from torchvision.utils import save_image
from net import *
import os
import utils
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from custom_adam import LREQAdam
from dataloader import *
from tqdm import tqdm
from dlutils.pytorch import count_parameters
import dlutils.pytorch.count_parameters as count_param_override
from tracker import LossTracker
from model import Model
from launcher import run
from defaults import get_cfg_defaults
import lod_driver
from PIL import Image
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import pickle

Z_path = ""
output_size = 32
total = 25
start_offset = 0

class ExtentedGenerator(torch.nn.Module):
    def __init__(self, generator, output_size=32):
        super().__init__()
        self.generator = generator
        self.avg = torch.nn.AdaptiveAvgPool2d(output_size=output_size)
        
        self.affine_dict = {}
        self.grads = {}
        self._register_hooks()
    
    def _register_hooks(self):
        layer  = 0
        for module in self.generator.decoder.decode_block:
            module.style_1.register_full_backward_hook(self.save_grad)
            module.style_2.register_full_backward_hook(self.save_grad)
            self.affine_dict[module.style_1] = "layer%d_style_1"%layer
            self.affine_dict[module.style_2] = "layer%d_style_2"%layer
            layer += 1
        
    def save_grad(self, module, grad_input, grad_output):
        name = self.affine_dict[module]
        if name in self.grads.keys():
            self.grads[name].append(grad_output[0].cpu())
        else:
            self.grads[name] = [grad_output[0].cpu()]
    
    def run(self, z_in , lod, blend):
        img_output = self.avg(self.generator.generate( lod, blend ,z_in, count=1, mixing= True))
        return img_output.flatten()
    
    def reset_grads(self):
        self.grads = {}
    
    @property
    def gradient_masks(self):
        style_grads = []
        for name, grads in self.grads.items():
            style_grads.append(torch.cat(grads, dim=0))
        
        return style_grads
        return torch.cat(style_grads[::-1], dim=1)

def GetGradientMaps(cfg, logger, local_rank, world_size, distributed):
    model = Model(
            startf=cfg.MODEL.START_CHANNEL_COUNT,
            layer_count=cfg.MODEL.LAYER_COUNT,
            maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
            latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
            dlatent_avg_beta=cfg.MODEL.DLATENT_AVG_BETA,
            style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
            mapping_layers=cfg.MODEL.MAPPING_LAYERS,
            channels=cfg.MODEL.CHANNELS,
            generator=cfg.MODEL.GENERATOR,
            encoder=cfg.MODEL.ENCODER,
            z_regression=cfg.MODEL.Z_REGRESSION
        )
    model.cuda(local_rank)

    decoder = model.decoder
    encoder = model.encoder
    mapping_d = model.mapping_d
    mapping_f = model.mapping_f
    dlatent_avg = model.dlatent_avg

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        'discriminator': encoder,
        'generator': decoder,
        'mapping_tl': mapping_d,
        'mapping_fl': mapping_f,
        'dlatent_avg': dlatent_avg
        }


    checkpointer = Checkpointer(cfg,
                                model_dict,
                                logger=logger,
                                save=local_rank == 0)

    extra_checkpoint_data = checkpointer.load()

    extendedGenerator = ExtentedGenerator(model , output_size = output_size)

    rnd = np.random.RandomState(5)

    lod = cfg.DATASET.MAX_RESOLUTION_LEVEL-2
    blend_factor = 1

    num_once = 1 

    Z = np.load(Z_path)

    gradient_maps_all_images = [[]]*18
    for i in range(18):
        gradient_maps_all_images[i] = [[]]*total
        
    for ctr in tqdm(range(total//num_once)):
        latents = Z[start_offset+ctr].reshape([num_once , -1])
        samplez = torch.tensor(latents).float().cuda() #z

        img = extendedGenerator.run(samplez , lod=lod , blend=blend_factor)
        
        for pixel in img:
            pixel.backward(retain_graph=True)

        gradient_maps = extendedGenerator.gradient_masks
        # gradient_maps_all_images.append(gradient_maps)
        
        for i in range(18):
            gradient_maps_all_images[i][ctr]=(gradient_maps[i].permute(1,0))
            # print(v.shape)

        extendedGenerator.reset_grads()
        model.zero_grad()
    
    print(len(gradient_maps_all_images))
    print(len(gradient_maps_all_images[0]))
    for i in range(18):
        gradient_maps_all_images[i] = np.stack(gradient_maps_all_images[i])

    with open("gradient_mask_%d/%d"%(output_size,start_offset), 'wb') as handle:
        pickle.dump(gradient_maps_all_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--Z_path',default='npy/ffhq_NT/Z_100.npy',type=str,help='path to Z vectors')
    parser.add_argument('--output_size',default=32,type=int,help='output size of the gradients')
    parser.add_argument("--sindex" , default=0 , type = int, help="Starting index in Z file")
    parser.add_argument("--num_per" , default=25 , type = int, help="Number of Samples")

    opt = parser.parse_args()

    Z_path = opt.Z_path
    output_size = opt.output_size
    total = opt.num_per
    start_offset = opt.sindex

    cfg = get_cfg_defaults()
    gpu_count = torch.cuda.device_count()
    run(GetGradientMaps, get_cfg_defaults(), description='StyleGAN', default_config='configs/ffhq.yaml',
        world_size=gpu_count)

