import torch
import numpy as np
import pathlib
import pandas as pd
import pickle
from training.networks_stylegan2 import SynthesisLayer
import kornia
import utils.image as imu
import matplotlib.pyplot as plt
from pprint import pprint
from test_iin import load_generator

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

# G = load_generator()
# t = torch.zeros(1).cuda()

# z1, z2 = [torch.randn(1, 512).cuda() for _ in range(2)]
# w1, w2 = G.mapping(z1, None), G.mapping(z2, None)

# wt1 = w1.lerp(w2, t.unsqueeze(1).unsqueeze(2))
# wt2 = w1.lerp(w2, t.unsqueeze(1).unsqueeze(2) + 1e-4)

# img1, img2 = G.synthesis(wt1, noise_mode='const'), G.synthesis(wt2, noise_mode='const')

img1, img2 = torch.load("./stylegan3/test.pt")[0], torch.load("./stylegan3/test.pt")[2]
imu.save_tensor_image("img1.png", img1, drange=(-1, 1))
imu.save_tensor_image("img2.png", img2, drange=(-1, 1))




