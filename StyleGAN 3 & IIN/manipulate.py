import numpy as np 
import torch
import matplotlib.pyplot as plt
import pickle 
import pandas as pd
from get_latent_codes import load_model, get_affine_layers
import pathlib
from utils import image as imu
import imageio
from torchvision.utils import make_grid, save_image


STYLE_DICT = {
    "input": (0, 4),
    "L0": (4, 516),
    "L1": (516, 1028),
    "L10": (4626, 4754),
    "L11": (4754, 4835),
    "L12": (4835, 4886),
    "L13": (4886, 4918),
    "L14": (4918, 4950),
    "L2": (1028, 1540),
    "L3": (1540, 2052),
    "L4": (2052, 2564),
    "L5": (2564, 3076),
    "L6": (3076, 3588),
    "L7": (3588, 4100),
    "L8": (4100, 4423),
    "L9": (4423, 4626),
}

def index_to_layer_channel(findex):
        if findex <= 4:
            return "input", findex
        for layer, in_range in STYLE_DICT.items():
            if findex in range(*in_range):
                return layer, findex - in_range[0]


def lp2istr(x):
    return str(x[0])+'_'+str(x[1])

def w_to_style(G, ws):
    affines = get_affine_layers(G)
    styles = []
    for layer in affines:
        styles.append(layer(ws))
    return torch.cat(styles, dim=1)

class Manipulator:
    
    
    def __init__(self, G, w_path, style_path, semantic_path, label_path, attribute_path):
        self.positive_bank = 1000
        self.num_pos = 10 #example
        self.num_m = 10 #number of output
        self.threshold1 = 0.5 #pass this ratio
        self.threshold2 = 0.25 #gap between first and second
        self.w = np.load(w_path)
        self.styles = np.load(style_path)
        
        self.styles_mean = np.mean(self.styles, axis=0)
        self.styles_std = np.std(self.styles, axis=0) 
        
        self.load_semantic(semantic_path, label_path)
        self.results = pd.read_csv(attribute_path)
        
        self.G = G

    
    def load_semantic(self, semantic_path, label_path):
        with open(semantic_path, 'rb') as handle:
            all_semantic_top = pickle.load(handle)
            
        self.all_semantic_top2 = np.concatenate(all_semantic_top)
        self.num_semantic = self.all_semantic_top2.shape[1]
        
        tmp = pd.read_csv(label_path)
        self.label = tmp['names']
        
        print(self.label)
        
    
    
    def get_rank(self,target_index):
        top_sum=self.all_semantic_top2[:,target_index].sum(axis=1)
        
        tmp=list(np.arange(self.num_semantic))
        for i in target_index:
            tmp.remove(i)
        tmp=self.all_semantic_top2[:,tmp] #all the rest semantic 
        second_max = tmp.max(axis=1)
        
        select1 = top_sum > self.threshold1
        select2 = top_sum-second_max > self.threshold2
        
        select = np.logical_and(select1,select2)
        findex = np.arange(len(select))[select]
        
        top_sum_index = top_sum[findex].argsort()[::-1]
        return findex[top_sum_index]
        
    
    def all_check(self, bname, positive=True):
        
        tmp_save = self.num_pos
        self.num_pos = self.positive_bank
        
        positive_train = self.simulate_input(bname, positive)
        s2n_ratios, indices = self.get_component(positive_train)
        
        self.num_pos = tmp_save
        
        return s2n_ratios, indices
    
    def simulate_input(self, bname, positive=True):
        tmp_indexs = self.results[bname].argsort()
        if positive:
            tmp=tmp_indexs[:self.positive_bank]
        else:
            tmp = tmp_indexs[-self.positive_bank:]
        positive_indexs = np.random.choice(tmp, size=self.num_pos, replace=False)
        
        device = next(self.G.parameters()).device
        tmp = torch.from_numpy(self.w[positive_indexs]).to(device)
        positive_train = w_to_style(self.G, tmp)
        return positive_train.cpu().numpy()
    
    def signal_to_noise(self, positive_train):
        normalize_positive = (positive_train - self.styles_mean)/self.styles_std
        feature_mean = np.abs(normalize_positive.mean(axis=0))
        feature_std = normalize_positive.std(axis=0)
        feature_s2n = feature_mean/feature_std
        return feature_s2n
    
    def get_component(self, positive_train):
        feature_s2n = self.signal_to_noise(positive_train)
        feature_index = feature_s2n.argsort()
        findex = feature_index[::-1]
                               
        return feature_s2n[findex], findex
        

    

if __name__ == "__main__":
    
    device = "cuda"
    semantic_path = "/cluster/scratch/ssadat/dl-project-outputs/semantic_top_32"
    w_path = "/cluster/scratch/ssadat/dl-project-outputs/latents/source-ws.npy"
    style_path = "/cluster/scratch/ssadat/dl-project-outputs/latents/source-styles.npy"
    attribute_path = "/cluster/scratch/ssadat/dl-project-outputs/attribute"
    label_path = "./StyleSpace/npy/ffhq/label"
    scratch_path = pathlib.Path("/cluster/scratch/ssadat")
    source_zlatents = torch.load(scratch_path.joinpath("dl-project-outputs/latents/source-zs.pt")).to(device)
    model_path = scratch_path.joinpath("stgan-models/stylegan3-t-ffhq-1024x1024.pkl")
    
    attr_name = "19-eyeglasses"
    z_index = 0
    alpha = 5
    
    G, _ = load_model(model_path)   
    manip = Manipulator(
        G=G, w_path=w_path, 
        style_path=style_path, 
        semantic_path=semantic_path, 
        label_path=label_path,
        attribute_path=attribute_path
    )
    
    # feature_s2n, index = manip.all_check(attr_name)
    # index = index[1]
    index = manip.get_rank((7,))[0]
    layer, channel = index_to_layer_channel(index)
    
    orig_z = source_zlatents[z_index:z_index+1].to(device)
    original_image = G(orig_z, None, truncation_psi=0.5)
    imu.save_tensor_image("orig.png", original_image, drange=(-1, 1))
    
    
    image_list = []
    for idx, strength in enumerate(np.linspace(-alpha, alpha, 4)):
        
        G, _ = load_model(model_path)
        
        def affine_hook(layer, input, output):
            output[:, channel] = output[:, channel] + strength * manip.styles_std[index]
            return output
        
        for name, module in G.named_modules():
            if "affine" in name:
                block_name = name.split(".")[1]
                layer_name = block_name.split("_")[0]
                if layer == layer_name:
                    module.register_forward_hook(affine_hook)
    
        img = G(orig_z, None, truncation_psi=0.5)
        # img = imu.from_torch(img, drange=(-1, 1))
        image_list.append(((img+1)/2).cpu().squeeze())
    save_image(make_grid(image_list, nrow=4), "test.png")
        
    
        
    
    