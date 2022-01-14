# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Perceptual Path Length (PPL) from the paper "A Style-Based Generator
Architecture for Generative Adversarial Networks". Matches the original
implementation by Karras et al. at
https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py"""

import copy
import numpy as np
import torch
# import metric_utils

import torch.utils.data
from torchvision.utils import save_image
from net import *
from checkpointer import Checkpointer
from dataloader import *
from model import Model
from launcher import run
from defaults import get_cfg_defaults
from PIL import Image
import pickle

from tqdm import tqdm


cfg = get_cfg_defaults()
#----------------------------------------------------------------------------

# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d

#----------------------------------------------------------------------------

def load_model(cfg, logger, local_rank, world_size, distributed):
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

    return model

class PPLSampler(torch.nn.Module):
    def __init__(self, G, G_kwargs, epsilon, space, sampling, crop, vgg16):
        assert space in ['z', 'w']
        assert sampling in ['full', 'end']
        super().__init__()
        self.G = copy.deepcopy(G)
        self.G_kwargs = G_kwargs
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.crop = crop
        self.vgg16 = copy.deepcopy(vgg16)

        self.lod = cfg.DATASET.MAX_RESOLUTION_LEVEL-2
        self.blend_factor = 1

    def forward(self):
        # Generate random latents and interpolation t-values.
        t = torch.rand([2], device="cuda") * (1 if self.sampling == 'full' else 0)
        z0, z1 = torch.randn([2, 512], device="cuda")#.chunk(2)

        # print(z0.reshape([1,-1]).shape)

        # Interpolate in W or Z.
        if self.space == 'w':
            w0 = self.G.mapping_f(z0.reshape([1,-1]))
            w1 = self.G.mapping_f(z1.reshape([1,-1]))
            wt0 = w0.lerp(w1, t.unsqueeze(1).unsqueeze(2))
            wt1 = w0.lerp(w1, t.unsqueeze(1).unsqueeze(2) + self.epsilon)
        else: # space == 'z'
            zt0 = slerp(z0, z1, t.unsqueeze(1))
            zt1 = slerp(z0, z1, t.unsqueeze(1) + self.epsilon)
            w0 = self.G.mapping_f(z0.reshape([1,-1]))
            w1 = self.G.mapping_f(z1.reshape([1,-1]))

        # Randomize noise buffers.
        for name, buf in self.G.named_buffers():
            if name.endswith('.noise_const'):
                buf.copy_(torch.randn_like(buf))

        # Generate images.
        img = self.G.decoder(torch.cat([wt0,wt1]),self.lod, self.blend_factor, False)

        # Center crop.
        if self.crop:
            assert img.shape[2] == img.shape[3]
            c = img.shape[2] // 8
            img = img[:, :, c*3 : c*7, c*2 : c*6]

        # Downsample to 256x256.
        factor = 1024 // 256
        if factor > 1:
            img = img.reshape([-1, img.shape[1], img.shape[2] // factor, factor, img.shape[3] // factor, factor]).mean([3, 5])

        # Scale dynamic range from [-1,1] to [0,255].
        img = (img + 1) * (255 / 2)

        # Evaluate differential LPIPS.
        lpips_t0, lpips_t1 = self.vgg16(img, resize_images=False, return_lpips=True).chunk(2)
        dist = (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon ** 2
        return dist

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    # assert 0 <= rank < num_gpus
    # key = (url, device)
    # if key not in _feature_detector_cache:
    #     is_leader = (rank == 0)
    #     if not is_leader and num_gpus > 1:
    #         torch.distributed.barrier() # leader goes first
    #     with open_url(url, verbose=(verbose and is_leader)) as f:
    #         _feature_detector_cache[key] = pickle.load(f).to(device)
    #     if is_leader and num_gpus > 1:
    #         torch.distributed.barrier() # others follow
    # return _feature_detector_cache[key]

    with open("vgg16.pkl", "rb") as f:
        vgg16 = pickle.load(f)
    
    return vgg16

def compute_ppl(opts, num_samples, epsilon, space, sampling, crop, batch_size):
    vgg16_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
    vgg16 = get_feature_detector(vgg16_url, num_gpus=opts["num_gpus"], rank=opts["rank"], verbose=True)

    # Setup sampler and labels.
    sampler = PPLSampler(G=opts["G"], G_kwargs=None, epsilon=epsilon, space=space, sampling=sampling, crop=crop, vgg16=vgg16)
    sampler.eval().requires_grad_(False).to("cuda")
    
    # Sampling loop.
    dist = []
    # progress = opts.progress.sub(tag='ppl sampling', num_items=num_samples)
    for batch_start in tqdm(range(0, num_samples, batch_size * opts["num_gpus"])):
        # progress.update(batch_start)
        x = sampler()
        for src in range(opts["num_gpus"]):
            y = x.clone()
            if opts["num_gpus"] > 1:
                torch.distributed.broadcast(y, src=src)
            dist.append(y)
    # progress.update(num_samples)

    # Compute PPL.
    if opts["rank"] != 0:
        return float('nan')
    dist = torch.cat(dist)[:num_samples].cpu().numpy()
    lo = np.percentile(dist, 1, interpolation='lower')
    hi = np.percentile(dist, 99, interpolation='higher')
    ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
    return float(ppl)

#----------------------------------------------------------------------------

def Execute(cfg, logger, local_rank, world_size, distributed):
    model = load_model(cfg, logger, local_rank, world_size, distributed)

    opts = {}

    opts["G"] = model
    opts["G_kwargs"] = None
    opts["num_gpus"] = torch.cuda.device_count()
    opts["rank"] = 0

    ppl = compute_ppl(opts, num_samples=50000, epsilon=10e-4 , space= "w" , sampling="end", crop=False , batch_size=2)
    print(ppl)

if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    run(Execute, get_cfg_defaults(), description='StyleGAN', default_config='configs/ffhq.yaml',
        world_size=gpu_count)