import pathlib
import pickle
import zipfile
from pprint import pprint
from time import process_time
import tempfile
import gc

import cv2
import kornia
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import utils.image as imu
from training.networks_stylegan3 import SynthesisLayer
from utils.jacobian import ExtentedGenerator
import click
from utils import distributed_utils as dst
from torch_utils import custom_ops


scratch_path = pathlib.Path("/cluster/scratch/ssadat")
model_path = scratch_path.joinpath("stgan-models/stylegan3-t-ffhq-1024x1024.pkl")
ngpus = 2


def tqdm_setup(iterable, should_enumerate=False):
    if should_enumerate:
        return tqdm(enumerate(iterable), total=len(iterable), leave=False)
    else:
        return tqdm(iterable, total=len(iterable), leave=False)


def load_model(model_path, device="cuda"):
    with open(model_path, "rb") as f:
        models = pickle.load(f)
    return models["G_ema"].to(device), models["D"].to(device)


def get_affine_layers(G):
    affine_layers = []
    for name, module in G.synthesis.named_children():
        affine_layers.append(module.affine)
    return affine_layers


def get_latent_vectors():
    n_total = 50000
    seed = 0
    rnd = np.random.RandomState(seed)
    device = "cuda"
    batch_size = 16
    
    G, _ = load_model(model_path, device)
    
    source_zlatents = torch.zeros(n_total, G.z_dim)
    source_wlatents = torch.zeros(n_total, G.w_dim)

    
    for i in tqdm_setup(range(0, n_total, batch_size)):
        
        z = torch.from_numpy(rnd.randn(batch_size, G.z_dim)).to(device)
        w = G.mapping(z, None)
        
        source_zlatents[i:i+batch_size, :] = z.cpu()
        source_wlatents[i:i+batch_size, :] = w[:,0,:].cpu()

    torch.save(source_zlatents, scratch_path.joinpath("dl-project-outputs/latents/source-zs.pt"))
    torch.save(source_wlatents, scratch_path.joinpath("dl-project-outputs/latents/source-ws.pt"))
    
    
    
def get_style_vectors():
    
    device = "cuda"
    n_total = 50000
    batch_size = 16
    
    scratch_path = pathlib.Path("/cluster/scratch/ssadat")
    model_path = scratch_path.joinpath("stgan-models/stylegan3-t-ffhq-1024x1024.pkl")
    
    G, _ = load_model(model_path, device)
    
    source_zlatents = torch.load(scratch_path.joinpath("dl-project-outputs/latents/source-zs.pt")).to(device)
    source_wlatents = torch.load(scratch_path.joinpath("dl-project-outputs/latents/source-ws.pt")).to(device)
    
    affine_layers = get_affine_layers(G)
    
    source_styles = []
    for idx in tqdm_setup(range(0, n_total, batch_size)):
        w = source_wlatents[idx:idx+batch_size, :]
        style_vector = []
        for layer in affine_layers:
             style_vector.append(layer(w))
        source_styles.append(torch.cat(style_vector, dim=1).cpu())
    
    source_styles = torch.cat(source_styles, dim=0)
    torch.save(source_styles, scratch_path.joinpath("dl-project-outputs/latents/source-styles.pt"))


def get_images():
    
    device = "cuda"
    n_total = 50000
    
    G, _ = load_model(model_path, device)
    
    source_zlatents = torch.load(scratch_path.joinpath("dl-project-outputs/latents/source-zs.pt")).to(device)
    save_path = scratch_path.joinpath("dl-project-outputs/stylegan3-fake-images.npy")
    
    all_images = []
    for idx in tqdm_setup(range(0, n_total)):
        img = G(source_zlatents[idx:idx+1], None)
        img = imu.change_drange(img, dlow=-1, dhigh=1)
        img = imu.to_cv2(imu.to_uint8(kornia.tensor_to_image(img)))
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LANCZOS4)
        img = imu.from_cv2(img)
        all_images.append(img)
    all_images = np.stack(all_images, axis=0)
    np.save(save_path, all_images) 
    

def get_grads(rank, temp_dir, ngpus=8):
    
    dst.init_process(rank, ngpus, temp_dir)
    
    size = 13
    start_index = rank * size
    output_size = 4
    
    G, _ = load_model(model_path, dst.get_device())
    G.requires_grad_(True)
    
    extended_G = ExtentedGenerator(G, output_size=output_size)
    
    source_zlatents = torch.load(scratch_path.joinpath("dl-project-outputs/latents/source-zs.pt")).to(dst.get_device())[start_index:start_index+size]
    
    for idx, z_in in tqdm_setup(source_zlatents, should_enumerate=True):
        torch.autograd.functional.jacobian(extended_G.run, z_in[None])
        # torch.save(extended_G.gradient_masks.detach().cpu(), scratch_path.joinpath(f"dl-project-outputs/grads/style-grads-120-{rank}-{idx}.pt"))
        extended_G.reset_grads()

    # all_grads = torch.stack(all_grads, dim=0)
    
def main():
    get_latent_vectors()
    get_style_vectors()
    get_images()
    with tempfile.TemporaryDirectory() as temp_dir:
        torch.multiprocessing.spawn(fn=get_grads, args=[temp_dir, ngpus], nprocs=ngpus, start_method="spawn")
    
if __name__ == "__main__":
    main()
