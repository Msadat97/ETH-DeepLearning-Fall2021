import pickle
from pprint import pprint
import numpy as np 
import torch
from einops import rearrange
import utils.image as imu

with open("/cluster/scratch/ssadat/stgan-models/stylegan3-t-ffhq-1024x1024.pkl", 'rb') as f:
    models = pickle.load(f)

G = models["G_ema"]

style_dict = {}
for name, module in G.synthesis.named_modules():
    # print(name)
    if "affine" in name:
        module_info = name.split(".")[0]
        if module_info == "input":
            style_dict["Input"] = 4
        else:
            layer = module_info.split("_")[0]
            style_dict[layer] = module.out_features

grads = torch.load("/cluster/scratch/ssadat/dl-project-outputs/style-grads-total.pt").numpy()

index_range = {"input":(0, 4), "L0":(4, 4 + style_dict["L0"])}
for layer in range(1, 15):
    length = style_dict[f"L{layer}"]
    start = index_range[f"L{layer-1}"][1]
    index_range[f"L{layer}"] = (start, start + length)

new_grads = []
for layer, (start, end) in index_range.items():
    grads_ = grads[start:end, :, :]
    grads_ = rearrange(grads_, "style image pixel -> image style pixel")
    new_grads.append(grads_)

with open("/cluster/scratch/ssadat/dl-project-outputs/grads-correct-shape.pkl", "wb") as f:
    pickle.dump(new_grads, f)