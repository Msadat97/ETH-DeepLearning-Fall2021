import torch
from collections import defaultdict

class ExtentedGenerator(torch.nn.Module):
    def __init__(self, generator, output_size=32):
        super().__init__()
        self.generator = generator
        self.avg = torch.nn.AdaptiveAvgPool2d(output_size=output_size)
        
        self.affine_dict = {}
        self.grads = defaultdict(list)
        self._register_hooks()
    
    def _register_hooks(self):
        for name, module in self.generator.synthesis.named_children():
            module.affine.register_full_backward_hook(self.save_grad)
            self.affine_dict[module.affine] = name
    
    def save_grad(self, module, grad_input, grad_output):
        name = self.affine_dict[module]
        self.grads[name].append(grad_output[0].detach().cpu())
    
    def run(self, z_in):
        img_output = self.avg(self.generator(z_in, None))
        return img_output.flatten()
    
    def reset_grads(self):
        self.grads = defaultdict(list)
    
    @property
    def gradient_masks(self):
        style_grads = []
        for name, grads in self.grads.items():
            style_grads.append(torch.cat(grads, dim=0))
        return torch.cat(style_grads[::-1], dim=1)
