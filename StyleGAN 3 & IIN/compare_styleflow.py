from utils.model import load_model
import pathlib
import pickle
import torch
import utils.image as imu


def manipulate_stgan2(layer_name, channel, amount):
    scratch_path = pathlib.Path("/cluster/scratch/ssadat")
    model_path = scratch_path.joinpath("stgan-models/stylegan2-ffhq-1024x1024.pkl") 
    with open("/cluster/home/ssadat/dl-project/StyleFlow/data/sg2latents.pickle", 'rb') as f:
        ws = pickle.load(f)["Latent"]
    G, _ = load_model(model_path)
    latent = torch.from_numpy(ws[2]).cuda()
    
    img = G.synthesis(latent)
    imu.save_tensor_image("./assets/stgan2-original.png", img, drange=(-1, 1))
    
    def affine_hook(layer, input, output):
        print(output[:, channel])
        output[:, channel] = output[:, channel] + amount
        return output
        
    for name, module in G.named_modules():
        if name == layer_name:
            module.register_forward_hook(affine_hook)
    
    img = G.synthesis(latent)
    imu.save_tensor_image("./assets/stgan2-editted.png", img, drange=(-1, 1))

def main():
    attr_dict = {
        "gender": {"layer_name": "synthesis.b32.conv1.affine", "channel": 6, "amount": 8},
        "sideburns": {"layer_name": "synthesis.b64.conv1.affine", "channel": 237, "amount": -5},
    }
    manipulate_stgan2(**attr_dict["gender"])

if __name__ == "__main__":
    main()