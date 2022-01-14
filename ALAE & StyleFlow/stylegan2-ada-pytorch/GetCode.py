import numpy as np
import torch
from PIL import Image
import pickle

import dnnlib
from tqdm import tqdm

def GetCode():

    with open('models/ffhq.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    
    z = torch.randn([1, G.z_dim]).cuda()    # latent codes
    # print(G)

    encoder = G.mapping
    
    styleA = []

    bs = [G.synthesis.b4, G.synthesis.b8,G.synthesis.b16,G.synthesis.b32,G.synthesis.b64,G.synthesis.b128,G.synthesis.b256,G.synthesis.b512,G.synthesis.b1024]
    for m in bs:

        if m.in_channels != 0: # First block has no conv0
            styleA.append(m.conv0.affine)
        styleA.append(m.conv1.affine)
        styleA.append(m.torgb.affine)

    IMGS = []
    W= []
    S=[]
    Z=[]

    rnd = np.random.RandomState(5)

    total = 10000
    num_once = 1
    for _ in tqdm(range(total//num_once)):
        z = rnd.randn(num_once, G.z_dim)
        z = torch.tensor(z).float().cuda() #z.cuda()
        w = encoder(z,None)[:,0,:].reshape([num_once, 512])

        s = []

        for m in styleA:
            s.append(m(w).cpu().numpy())
        
        s = np.hstack(s)
        
        img = G(z, None)[0].permute(1,2,0).cpu().numpy()
        img = img * 0.5 + 0.5
        img = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), 'RGB').resize([256,256])
        img = np.asarray(img)

        IMGS.append(img)
        S.append(s[0])
        W.append(w.cpu().numpy())
        Z.append(z.cpu().numpy())
    
    Z = np.vstack(Z)
    W = np.vstack(W)
    S = np.asarray(S)
    IMGS = np.asarray(IMGS)

    print(Z.shape , W.shape, S.shape, IMGS.shape)
    np.save("npy/ffhq_NT/Z_%d.npy"%total , Z)
    np.save("npy/ffhq_NT/W_%d.npy"%total , W)
    np.save("npy/ffhq_NT/S_Flat_%d.npy"%total , S)
    np.save("npy/ffhq_NT/images_%d.npy"%total,IMGS)

if __name__ == "__main__":
    GetCode()