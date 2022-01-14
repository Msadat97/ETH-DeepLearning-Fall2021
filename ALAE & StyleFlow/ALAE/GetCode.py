# Copyright 2019-2020 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python3
import torch.utils.data
from torchvision.utils import save_image
from net import *
import os
import utils
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from custom_adam import LREQAdam
from dataloader import *
from dlutils.pytorch import count_parameters
import dlutils.pytorch.count_parameters as count_param_override
from tracker import LossTracker
from model import Model
from launcher import run
from defaults import get_cfg_defaults
import lod_driver
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pickle
import argparse

cfg = get_cfg_defaults()

save_path = ""
num_samples = 100
resize = None

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

    # model_dict['discriminator_s'] = model_s.encoder
    # model_dict['generator_s'] = model_s.decoder
    # model_dict['mapping_tl_s'] = model_s.mapping_d
    # model_dict['mapping_fl_s'] = model_s.mapping_f

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                logger=logger,
                                save=local_rank == 0)

    extra_checkpoint_data = checkpointer.load()

    return model

def GetSFlat(cfg, logger, local_rank, world_size, distributed):
    model = load_model(cfg, logger, local_rank, world_size, distributed)

    rnd = np.random.RandomState(5)

    # print(model)

    lod = cfg.DATASET.MAX_RESOLUTION_LEVEL-2
    blend_factor = 1

    total = num_samples
    num_once = 1
    Z = []
    W = []
    S = []
    IMGS = []
    style_A = []

    for m in model.decoder.decode_block:
        style_A.append(m.style_1)
        style_A.append(m.style_2)


    for _ in tqdm(range(total//num_once)):
        latents = rnd.randn(num_once, cfg.MODEL.LATENT_SPACE_SIZE) 
        samplez = torch.tensor(latents).float().cuda() #z
        img = model.generate(lod, blend_factor, samplez, count=1, mixing=True)[0].cpu().detach().numpy()
        img = img.transpose(1, 2, 0)
        img = img * 0.5 + 0.5
        img = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), 'RGB')

        if resize is not None:
            img = img.resize([resize , resize])

        img = np.asarray(img)

        IMGS.append(img)

        # img.save("sampled.png")
        # plt.imshow(img)
        # plt.show()

        styles = model.mapping_f(samplez)[:, 0]
        
        s_flat = [] 
        for m in style_A:
            s = m(styles)
            s_flat.append(s.cpu().detach().numpy()[0])

        s_flat = np.hstack(s_flat)
        S.append(s_flat)
        Z.append(samplez.cpu().detach().numpy())
        W.append(styles.cpu().detach().numpy())
    
    # print(Z.shape , W.shape, S.shape , IMGS.shape)

    Z = np.vstack(Z)
    W = np.vstack(W)
    S = np.asarray(S)
    IMGS = np.asarray(IMGS)

    Z_path = os.path.join(save_path , "Z_%d.npy"%total)
    W_path = os.path.join(save_path , "W_%d.npy"%total)
    S_path = os.path.join(save_path , "SFlat_%d.npy"%total)
    IMG_path = os.path.join(save_path , "IMG_%d.npy"%total)

    np.save(Z_path , Z)
    np.save(W_path, W)
    np.save(S_path , S)
    np.save(IMG_path,IMGS)

    print(Z.shape , W.shape, S.shape , IMGS.shape)

def GetS(cfg, logger, local_rank, world_size, distributed):
    model = load_model(cfg, logger, local_rank, world_size, distributed)

    rnd = np.random.RandomState(5)

    # print(model)

    lod = cfg.DATASET.MAX_RESOLUTION_LEVEL-2
    blend_factor = 1

    total = num_samples
    num_once = 1
    Z = []
    W = []
    S = []
    IMGS = []
    style_A = []

    for m in model.decoder.decode_block:

        style_A.append(m.style_1)
        style_A.append(m.style_2)

    style_A.reverse()

    S = [[]]*18

    for i in range(18):
        S[i] = [[]]*total

    for current_img in tqdm(range(total//num_once)):
        latents = rnd.randn(num_once, cfg.MODEL.LATENT_SPACE_SIZE) 
        samplez = torch.tensor(latents).float().cuda() #z
        img = model.generate(lod, blend_factor, samplez, count=1, mixing=True)[0].cpu().detach().numpy()
        img = img.transpose(1, 2, 0)
        img = img * 0.5 + 0.5
        img = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), 'RGB')#.resize([256,256]))

        if resize is not None:
            img = img.resize([resize , resize])

        img = np.asarray(img)

        IMGS.append(img)

        # img.save("sampled.png")
        # plt.imshow(img)
        # plt.show()

        styles = model.mapping_f(samplez)[:, 0]
        
        c = 0
        for m in style_A:
            s = m(styles)
            # print(s.shape)
            S[c][current_img]=s[0]
            c += 1

        Z.append(samplez.cpu().detach().numpy())
        W.append(styles.cpu().detach().numpy())

    Z = np.vstack(Z)
    W = np.vstack(W)
    IMGS = np.asarray(IMGS)
    print(Z.shape , W.shape, len(S[0])  , IMGS.shape)

    Z_path = os.path.join(save_path , "Z_%d.npy"%total)
    W_path = os.path.join(save_path , "W_%d.npy"%total)
    S_path = os.path.join(save_path , "S_%d.npy"%total)
    IMG_path = os.path.join(save_path , "IMG_%d.npy"%total)

    np.save(Z_path , Z)
    np.save(W_path, W)
    np.save(IMG_path,IMGS)

    with open(S_path , "wb") as f:
        pickle.dump(S, f , protocol=pickle.HIGHEST_PROTOCOL)

    return S

def GetCodeMS(dlatents):
        m=[]
        std=[]
        for i in tqdm(range(len(dlatents))):
            tmp= dlatents[i] 
            tmp_mean=tmp.mean(axis=0)
            tmp_std=tmp.std(axis=0)
            m.append(tmp_mean)
            std.append(tmp_std)
        return m,std

def GSMeanStd(cfg, logger, local_rank, world_size, distributed):

    dlatents = GetS(cfg, logger, local_rank, world_size, distributed)
    for i in range(18):
        dlatents[i] = torch.stack(dlatents[i])
    m,std=GetCodeMS(dlatents)
    save_tmp=[m,std]
    save_name='S_mean_std_%d'%len(dlatents[0])
    tmp=os.path.join(save_path , save_name)
    with open(tmp, "wb") as fp:
        pickle.dump(save_tmp, fp)

def GetActivation(cfg, logger, local_rank, world_size, distributed):
    model = load_model(cfg, logger, local_rank, world_size, distributed)

    rnd = np.random.RandomState(5)

    # print(model)

    lod = cfg.DATASET.MAX_RESOLUTION_LEVEL-2
    blend_factor = 1

    total = num_samples
    num_once = 1

    IMGS = []
    Acts = []

    for current_img in tqdm(range(total//num_once)):
        latents = rnd.randn(num_once, cfg.MODEL.LATENT_SPACE_SIZE) 
        samplez = torch.tensor(latents).float().cuda() #z
        # img = model.generate(lod, blend_factor, samplez, count=1, mixing=True)[0].cpu().detach().numpy()
        # img = img.transpose(1, 2, 0)
        
        styles = model.mapping_f(samplez)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])
        styles = s.repeat(1, model.mapping_f.num_layers, 1)
        
        img , activations = model.decoder.activations(styles , lod , noise = True)
        img = img[0].cpu().detach().numpy().transpose(1, 2, 0)
        img = img * 0.5 + 0.5
        img = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), 'RGB')#.resize([256,256]))

        if resize is not None:
            img = img.resize([resize , resize])

        img = np.asarray(img)

        active = activations[-5][0]
        active = np.average(active,axis = 0)

        active = (active - np.min(active) ) / (np.max(active) - np.min(active))

        IMGS.append(img)
        Acts.append(active)

        # fig , axs = plt.subplots(1 , 2)

        # axs[0].imshow(img)
        # axs[1].imshow(active)

        # plt.show()

    IMGS = np.asarray(IMGS)
    Acts = np.asarray(Acts)

    Act_path = os.path.join(save_path , "activations_%d_%d.npy"%(64,total))
    np.save(Act_path, Acts)

    fig , axs = plt.subplots(10 , 10)

    for i in range(10):
        for j in range(10):
            axs[i][j].imshow(Acts[i*10 + j])

    plt.show()

    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save_path',default='./npy/ffhq_NT/',type=str,help='path to save folder') 
    parser.add_argument('--num_samples',default=100,type=int,help='Number of images to generate') 
    parser.add_argument('--mode' , default="SFlat", type=str , choices=["SFlat", "S", "SMeanStd" , "Activations"], help="Mode for Style Space")
    parser.add_argument('--resize', default=None , type=int , help='The resolution of the output images')

    opt = parser.parse_args()

    save_path = opt.save_path
    num_samples = opt.num_samples
    resize = opt.resize

    gpu_count = torch.cuda.device_count()

    if opt.mode == "SFlat":
        run(GetSFlat, cfg, description='StyleGAN', default_config='configs/ffhq.yaml', world_size=gpu_count)
    elif opt.mode == "S":
        run(GetS, cfg, description='StyleGAN', default_config='configs/ffhq.yaml', world_size=gpu_count)
    elif opt.mode == "SMeanStd":
        run(GSMeanStd, cfg, description='StyleGAN', default_config='configs/ffhq.yaml', world_size=gpu_count)
    elif opt.mode == "Activations":
        run(GetActivation, cfg, description='StyleGAN', default_config='configs/ffhq.yaml', world_size=gpu_count)

