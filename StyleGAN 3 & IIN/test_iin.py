from iin.networks import VectorTransformer, FactorTransformer
from iin.dataset import TensorDataset
import torch
from get_latent_codes import load_model, tqdm_setup
import pathlib
import utils.image as imu
import pandas as pd
import numpy as np
import imageio
from copy import copy
from omegaconf import OmegaConf
import dnnlib
from torchvision.utils import make_grid, save_image


def vis(bname,suffix,out,rownames=None,colnames=None):
    num_images=out.shape[0]
    step=out.shape[1]
    
    if colnames is None:
        colnames=[f'Step {i:02d}' for i in range(1, step + 1)]
    if rownames is None:
        rownames=[str(i) for i in range(num_images)]
    
    
    visualizer = HtmlPageVisualizer(
      num_rows=num_images, num_cols=step + 1, viz_size=256)
    visualizer.set_headers(
      ['Name'] +colnames)
    
    for i in range(num_images):
        visualizer.set_cell(i, 0, text=rownames[i])
    
    for i in range(num_images):
        for k in range(step):
            image=out[i,k,:,:,:]
            visualizer.set_cell(i, 1+k, image=image)
            

def load_pos_neg_samples(config):
    attr = pd.read_csv(config.data.attribute_path)
    pos_sample = attr["01-smiling"] > 1
    neg_sample = attr["01-smiling"] < -1
    dataset = TensorDataset(config.data.source_path)
    pos_zs = dataset[pos_sample][0:1000]
    neg_zs = dataset[neg_sample][0:1000]
    return pos_zs, neg_zs

def load_iin_model(config):
    model = dnnlib.util.construct_class_by_name(**config.model)
    model.load_state_dict(torch.load(config.checkpoint_path))
    model = model.cuda()
    model.eval()
    return model


def load_generator():
    scratch_path = pathlib.Path("/cluster/scratch/ssadat")
    model_path = scratch_path.joinpath("stgan-models/stylegan3-t-ffhq-1024x1024.pkl")
    G, _ = load_model(model_path)
    return G


def manipulate_factors():
    
    config = OmegaConf.load("./iin/config_factor.yaml")
    
    model = load_iin_model(config)
    G = load_generator()
    
    rng = np.random.RandomState(1918)
    z1 = torch.from_numpy(rng.randn(1, 512)).cuda()
    rng = np.random.RandomState(400)
    z2 = torch.from_numpy(rng.randn(1, 512)).cuda()

    w1 = G.mapping(z1, None, truncation_psi=0.5)[:, 0]
    w2 = G.mapping(z2, None, truncation_psi=0.5)[:, 0]
        
    factors1, _ = model(w1)
    factors2, _ = model(w2)


    img1 = G.synthesis(w1.cuda().unsqueeze(1).repeat([1, G.num_ws, 1]))
    imu.save_tensor_image("test_inn1.png", img1, drange=(-1, 1))
    
    img2 = G.synthesis(w2.cuda().unsqueeze(1).repeat([1, G.num_ws, 1]))
    imu.save_tensor_image("test_inn2.png", img2, drange=(-1, 1))
    
    image_list = []
    for alpha in np.linspace(-2, 2, 4):
        factors_ = list(copy(factors1))
        factors_[3] = torch.lerp(factors1[1], factors2[1], alpha)
        w = model.reverse(factors_).flatten(start_dim=1)
        latent = w.unsqueeze(1).repeat([1, G.num_ws, 1])
        # latent[:, 5, :] = w
        img = G.synthesis(latent)
        image_list.append(((img+1)/2).cpu().squeeze())
    save_image(make_grid(image_list, nrow=4), "test.png")

def manipulate_wplus():

    config = OmegaConf.load("./iin/config.yaml")
    
    model = load_iin_model(config)
    G = load_generator()
    
    rng = np.random.RandomState(1918)
    z1 = torch.from_numpy(rng.randn(1, 512)).cuda()
    rng = np.random.RandomState(400)
    z2 = torch.from_numpy(rng.randn(1, 512)).cuda()

    w1 = G.mapping(z1, None, truncation_psi=0.5)
    w2 = G.mapping(z2, None, truncation_psi=0.5)
        
    factors1, _ = model(w1.flatten(start_dim=1))
    factors2, _ = model(w2.flatten(start_dim=1))


    img1 = G.synthesis(w1[:, 0].cuda().unsqueeze(1).repeat([1, G.num_ws, 1]))
    imu.save_tensor_image("test_inn1.png", img1, drange=(-1, 1))
    
    img2 = G.synthesis(w2[:, 0].cuda().unsqueeze(1).repeat([1, G.num_ws, 1]))
    imu.save_tensor_image("test_inn2.png", img2, drange=(-1, 1))

    image_list = []
    for alpha in np.linspace(-2, 2, 20):
        factors_ = list(copy(factors1))
        factors_[3] = torch.lerp(factors1[1], factors2[1], alpha)
        w = model.reverse(factors_).flatten(start_dim=1).reshape(1, 16, 512)
        # latent = w.unsqueeze(1).repeat([1, G.num_ws, 1])
        # latent[:, 5, :] = w
        img = G.synthesis(w)
        img = imu.from_torch(img, drange=(-1, 1))
        image_list.append(img)
    
    imageio.mimsave("interp.gif", image_list, fps=5)


if __name__ == "__main__":
    manipulate_factors()


