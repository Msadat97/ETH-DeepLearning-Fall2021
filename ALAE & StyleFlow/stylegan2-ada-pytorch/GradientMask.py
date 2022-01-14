import numpy as np
import torch
from PIL import Image
import pickle

import dnnlib
from tqdm import tqdm
from collections import defaultdict

class ExtentedGenerator(torch.nn.Module):
    def __init__(self, generator, output_size=32):
        super().__init__()
        self.generator = generator
        self.avg = torch.nn.AdaptiveAvgPool2d(output_size=output_size)
        
        self.affine_dict = {}
        self.grads = defaultdict(list)
        self.num_modules = 0
        self._register_hooks()
    
    def _register_hooks(self):
        for name, module in self.generator.synthesis.named_children():
            
            if module.in_channels != 0:
                module.conv0.affine.register_backward_hook(self.save_grad)
                self.num_modules += 1
                self.affine_dict[module.conv0.affine] = name + "_conv0"

            module.conv1.affine.register_backward_hook(self.save_grad)
            self.affine_dict[module.conv1.affine] = name + "_conv1"
            self.num_modules += 1

            module.torgb.affine.register_backward_hook(self.save_grad)
            self.affine_dict[module.torgb.affine] = name + "_torgb"
            self.num_modules += 1
    
    def save_grad(self, module, grad_input, grad_output):
        name = self.affine_dict[module]
        self.grads[name].append(grad_output[0].detach().cpu())
    
    def run(self, w_in):
        img_output = self.avg(self.generator.synthesis(w_in))
        return img_output.flatten()
    
    def reset_grads(self):
        self.grads = defaultdict(list)
    
    @property
    def gradient_masks(self):
        style_grads = []
        for name, grads in self.grads.items():
            style_grads.append(torch.cat(grads, dim=0))
        return style_grads
        # return torch.cat(style_grads[::-1], dim=1)


def GetGradients():

    with open('models/ffhq.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

    G.requires_grad_(True)

    # print(G)
    output_size = 32
    extended_G = ExtentedGenerator(G , output_size = output_size)

    rnd = np.random.RandomState(5)

    total = 25
    start_offset = 75

    w_flow = np.load("npy/ffhq_NT/W_flow_new.npy")

    gradient_maps_all_images = [[]]*extended_G.num_modules
    for i in range(extended_G.num_modules):
        gradient_maps_all_images[i] = [[]]*total

    for idx in tqdm(range(total)):
        w = w_flow[idx+start_offset]
        w = w.reshape([1,w.shape[0],w.shape[1]])
        w = torch.Tensor(w).cuda()
        img = extended_G.run(w)
        extended_G.reset_grads()

        for pixel in img:
            pixel.backward(retain_graph=True)

        gradient_maps = extended_G.gradient_masks
        # gradient_maps_all_images.append(gradient_maps)
        # print(gradient_maps[0].shape)
        for i in range(extended_G.num_modules):
            gradient_maps_all_images[i][idx]=(gradient_maps[i].permute(1,0))
            # print(v.shape)

        extended_G.reset_grads()
        G.zero_grad()

    print(len(gradient_maps_all_images))
    print(len(gradient_maps_all_images[0]))
    for i in range(extended_G.num_modules):
        gradient_maps_all_images[i] = np.stack(gradient_maps_all_images[i])

    with open("npy/ffhq_NT/gradient_mask_%d_%d"%(output_size,start_offset), 'wb') as handle:
        pickle.dump(gradient_maps_all_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return None

def GenerateImages():
    with open('models/ffhq.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

    G.requires_grad_(True)

    w_flow = np.load("npy/ffhq_NT/W_flow_new.npy")

    total = 100

    IMGS = []

    for idx in tqdm(range(total)):
        w = w_flow[idx]
        w = w.reshape([1,w.shape[0],w.shape[1]])
        w = torch.Tensor(w).cuda()
        img = G.synthesis(w)[0].cpu().detach().numpy()

        img = img.transpose(1, 2, 0)
        img = img * 0.5 + 0.5
        img = np.asarray(Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), 'RGB'))#.resize([256,256]))
        IMGS.append(img)

    IMGS = np.asarray(IMGS)
    print(IMGS.shape)

    np.save("npy/ffhq_NT/images_%d.npy"%total,IMGS)


if __name__ == "__main__":
    
    GenerateImages()