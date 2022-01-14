import numpy as np
from numpy.lib import load 
import torch
import pickle 
import pandas as pd
import argparse
import torch.utils.data
from torchvision.utils import save_image
from net import *
from checkpointer import Checkpointer
from dataloader import *
from model import Model
from launcher import run
from defaults import get_cfg_defaults
from PIL import Image
import imageio
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

cfg = get_cfg_defaults()
base_path = ""
w_file = ""
image_path = ""
semantic_file = ""
s_file = ""
attribute_file = ""


def lp2istr(x):
    return str(x[0])+'_'+str(x[1])

class MAdvance():
    
    def __init__(self,w_path, semantic_path, label_path,img_path):
        self.positive_bank=1000
        self.num_pos=10 #example
        self.num_m=10 #number of output
        self.threshold1=0.5 #pass this ratio
        self.threshold2=0.25 #gap between first and second
        
        self.w=np.load(w_path)

        self.fmaps=[32, 32, 64, 64, 128, 128,256 ,256, 512,512,1024,1024,1024,1024,1024,1024,1024,1024]
        self.fmaps.reverse()

        self.img_path = img_path
        self.dlatents,self.code_mean,self.code_std=self.LoadData(self.img_path)
        
        self.code_mean2=self.code_mean
        self.code_std2=self.code_std
        print(self.code_mean2.shape)

        self.bname = ""

        try:
            self.LoadSemantic(semantic_path, label_path)
        except FileNotFoundError:
            print('semantic_top_32 not exist')
        
        try:
            self.results=pd.read_csv(self.img_path+attribute_file)
        except FileNotFoundError:
            print('attribute not exist')

    def LoadSemantic(self,semantic_path, label_path):

        with open(semantic_path, 'rb') as handle:
            all_semantic_top = pickle.load(handle)

        all_semantic_top.reverse()
        self.all_semantic_top2=np.concatenate(all_semantic_top)
        self.num_semantic=self.all_semantic_top2.shape[1] #ignore low frequency area, bed 10
        
        tmp=pd.read_csv(label_path)
        self.label=tmp['names']
    
    def LoadData(self,img_path):
        tmp=os.path.join(img_path,s_file)
        dlatents = np.load(tmp)
        m = np.mean(dlatents,axis=0)
        std = np.std(dlatents,axis=0)

        return dlatents,m,std

    def AllCheck(self,positive=True):
        
        tmp_save=self.num_pos
        self.num_pos=self.positive_bank
        
        positive_train,_=self.SimulateInput(positive)
        index2,_=self.GetComponent(positive_train)
        
        self.num_pos=tmp_save
        lp_sort=pd.DataFrame(index2[:,-1])
        lp_sort.index=list(map(lp2istr, index2[:,:-1].astype(int)))
        
        return index2,lp_sort   

    def SimulateInput(self,positive=True):
        print('bname: '+str(self.bname))
        tmp_indexs=self.results[self.bname].argsort()
        if positive:
            tmp=tmp_indexs[:self.positive_bank]
        else:
            tmp=tmp_indexs[-self.positive_bank:]
        positive_indexs=np.random.choice(tmp,size=self.num_pos,replace=False)
        
        # tmp=self.w[positive_indexs] #only use 50 images
        # tmp=tmp[:,None,:]
        # w_plus=np.tile(tmp,(1,self.Gs.components.synthesis.input_shape[1],1))
        # tmp_dlatents=self.dlatents[positive_indexs]
        
        # positive_train=[self.dlatents[tmp] for tmp in positive_indexs]

        positive_train = self.dlatents[tmp]

        # positive_train = []
        # for i in range(len(self.dlatents)):
        #     tmp_latent = []
        #     for tmp in positive_indexs:
        #         tmp_latent.append(self.dlatents[i][tmp].cpu().detach())
        #     # print(len(tmp_latent))
        #     positive_train.append(np.stack(tmp_latent))
        return positive_train,positive_indexs

    def GetComponent(self,positive_train): #sort s2n, remove pg, 
        
        feature_s2n=self.S2N(positive_train)
        
        feature_index=feature_s2n.argsort()
        findex=feature_index[::-1] #index in concatenate form 
        
        l_p=self.GetLCIndex(findex)
        
        index2=np.zeros([len(l_p),3])
        index2[:,2]=feature_s2n[findex]
        index2[:,(0,1)]=l_p

        return index2,findex

    def S2N(self,positive_train):
        positive_train2=positive_train
        normalize_positive=(positive_train2-self.code_mean2)/self.code_std2
        
        feature_mean=np.abs(normalize_positive.mean(axis=0))
        feature_std=normalize_positive.std(axis=0)
        
        feature_s2n=feature_mean/feature_std
        return feature_s2n
   
    def GetRank(self,target_index, get_all = False):
        top_sum=self.all_semantic_top2[:,target_index].sum(axis=1)
        
        tmp=list(np.arange(self.num_semantic))
        for i in target_index:
            tmp.remove(i)
        tmp=self.all_semantic_top2[:,tmp] #all the rest semantic 
        second_max=tmp.max(axis=1)
        
        select1=top_sum>self.threshold1
        select2=top_sum-second_max>self.threshold2
        
        select=np.logical_and(select1,select2)
        findex=np.arange(len(select))[select]

        top_sum_index = top_sum[findex].argsort()[::-1]
        
        # print(top_sum[findex].shape)
        if get_all:
            return findex[top_sum_index], top_sum[findex]
        return findex[top_sum_index]
    
    def GetLCIndex(self,findex):
        l_p=[]
        cfmaps=np.cumsum(self.fmaps)
        for i in range(len(findex)):
            tmp_index=findex[i]
            tmp=tmp_index-cfmaps
            tmp=tmp[tmp>0]
            lindex=len(tmp)
            if lindex==0:
                cindex=tmp_index
            else:
                cindex=tmp[-1]
            
            if cindex ==self.fmaps[lindex]:
                cindex=0
                lindex+=1
            l_p.append([lindex,cindex])
        l_p=np.array(l_p)
        return l_p

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

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                logger=logger,
                                save=local_rank == 0)

    extra_checkpoint_data = checkpointer.load()

    return model

def Manipulate(cfg, logger, local_rank, world_size, distributed):
    
    model = load_model(cfg, logger, local_rank, world_size, distributed)

    rnd = np.random#.RandomState(5)
    lod = cfg.DATASET.MAX_RESOLUTION_LEVEL-2
    blend_factor = 1

    device = "cuda"
        
    semantic_path = os.path.join(base_path,semantic_file)
    w_path = os.path.join(base_path,w_file)
    label_path = "../StyleSpace-main/npy/ffhq/label"
    img_path = base_path
    M = MAdvance(w_path=w_path, semantic_path=semantic_path, label_path=label_path,img_path=img_path)

    latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE) 
    samplez = torch.tensor(latents).float().cuda()
    
    
    for i in range(11):
        index = M.GetRank((i, ))[1]
        layer, channel =M.GetLCIndex([index])[0]

        fig , axs = plt.subplots(1, 5)
    
        for idx, strength in enumerate([-10,-5,1,5,10]):
            
            model = load_model(cfg, logger, local_rank, world_size, distributed)
            
            def affine_hook(layer, input, output):
                output[:, channel] = output[:, channel] * strength
                return output
            
            l = layer // 2
            r = layer % 2

            if r == 0:
                model.decoder.decode_block[l].style_1.register_forward_hook(affine_hook)
            else:
                model.decoder.decode_block[l].style_2.register_forward_hook(affine_hook)
            
            img = model.generate(lod, blend_factor, samplez, count=1, mixing=False)[0].cpu().detach().numpy()
            img = img.transpose(1, 2, 0)
            img = img * 0.5 + 0.5
            img = np.asarray(Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), 'RGB'))#.resize([256,256]))
            axs[idx].imshow(img)
            axs[idx].set_xticks([])
            axs[idx].set_yticks([])

        plt.savefig("Manipulated_images/manipulated_%d.png"%i)

def PercentageLocalized(cfg, logger, local_rank, world_size, distributed):

    semantic_path = os.path.join(base_path,semantic_file)
    w_path = os.path.join(base_path,w_file)
    label_path = "../StyleSpace-main/npy/ffhq/label"
    img_path = base_path
    M = MAdvance(w_path=w_path, semantic_path=semantic_path, label_path=label_path,img_path=img_path)

    total_localized = 0

    for i in range(11):
        index , all_localized_channels = M.GetRank((i, ) , get_all= True)#[1]
        total_localized += len(all_localized_channels)

    print("Percentage Localized = %d / %d =  %f"%(total_localized,M.all_semantic_top2.shape[0],total_localized/M.all_semantic_top2.shape[0]))

def ManipulateReal(cfg, logger, local_rank, world_size, distributed):
    model = load_model(cfg, logger, local_rank, world_size, distributed)

    rnd = np.random#.RandomState(5)
    lod = cfg.DATASET.MAX_RESOLUTION_LEVEL-2
    blend_factor = 1

    semantic_path = os.path.join(base_path,semantic_file)
    w_path = os.path.join(base_path,w_file)
    label_path = "../StyleSpace-main/npy/ffhq/label"
    img_path = base_path
    M = MAdvance(w_path=w_path, semantic_path=semantic_path, label_path=label_path,img_path=img_path)

    latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE) 

    image_name = image_path
    img = imageio.imread(image_name)
    img = img / 255.0 
    img = img*2 - 1
    img = torch.Tensor(img).permute(2,0,1).reshape([1,3,img.shape[0],img.shape[1]])

    styles = model.encode(img , lod , blend_factor)[0][0]
    s = styles.view(styles.shape[0], 1, styles.shape[1])
    styles = s.repeat(1, model.mapping_f.num_layers, 1)

    for i in range(11):
        index = M.GetRank((i, ))[1]
        layer, channel =M.GetLCIndex([index])[0]

        fig , axs = plt.subplots(1, 5)
    
        for idx, strength in enumerate([-25,-10,1,10,25]):
            
            model = load_model(cfg, logger, local_rank, world_size, distributed)
            
            def affine_hook(layer, input, output):
                output[:, channel] = output[:, channel] * strength
                return output
            
            l = layer // 2
            r = layer % 2

            if r == 0:
                model.decoder.decode_block[l].style_1.register_forward_hook(affine_hook)
            else:
                model.decoder.decode_block[l].style_2.register_forward_hook(affine_hook)
            
            img = model.decoder.decode(styles ,lod , noise = False)[0].cpu().detach().numpy()
            img = img.transpose(1, 2, 0)
            img = img * 0.5 + 0.5
            img = np.asarray(Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), 'RGB'))#.resize([256,256]))
            axs[idx].imshow(img)
            axs[idx].set_xticks([])
            axs[idx].set_yticks([])

        plt.savefig("Manipulated_images/manipulated_%d.png"%i)
    
def AttributeDependent(cfg, logger, local_rank, world_size, distributed):
    model = load_model(cfg, logger, local_rank, world_size, distributed)

    rnd = np.random#.RandomState(5)
    lod = cfg.DATASET.MAX_RESOLUTION_LEVEL-2
    blend_factor = 1
        
    semantic_path = os.path.join(base_path,semantic_file)
    w_path = os.path.join(base_path,w_file)
    label_path = "../StyleSpace-main/npy/ffhq/label"
    img_path = base_path
    M = MAdvance(w_path=w_path, semantic_path=semantic_path, label_path=label_path, img_path=img_path)
    
    latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE) 
    samplez = torch.tensor(latents).float().cuda()

    for n in M.results.columns:
        M.bname=n
        fig , axs = plt.subplots(1 , 5)
        lp_candidate,lp_sort= M.AllCheck()
        
        layer,channel,_=lp_candidate[0]
        layer , channel = int(layer) , int(channel)

        for idx, strength in enumerate([-25,-10,1,10,25]):
                
            model = load_model(cfg, logger, local_rank, world_size, distributed)
            
            def affine_hook(layer, input, output):
                output[:, channel] = output[:, channel] * strength
                return output
                
            l = layer // 2
            r = layer % 2

            if r == 0:
                model.decoder.decode_block[l].style_1.register_forward_hook(affine_hook)
            else:
                model.decoder.decode_block[l].style_2.register_forward_hook(affine_hook)
                
            img = model.generate(lod, blend_factor, samplez, count=1, mixing=False)[0].cpu().detach().numpy()
            img = img.transpose(1, 2, 0)
            img = img * 0.5 + 0.5
            img = np.asarray(Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), 'RGB'))#.resize([256,256]))
            axs[idx].imshow(img)
            axs[idx].set_xticks([])
            axs[idx].set_yticks([])

            plt.savefig("Manipulated_images/Attribute_Dependent/manipulated_%s.png"%n)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--base_path',default='./npy/ffhq_NT/',type=str,help='path to data folder') 
    parser.add_argument('--mode' , default="Manipulate", type=str , choices=["Manipulate", "ManipulateReal", "AttributeDependent" , "PercentageLocalized"], help="Manipulation Mode")
    parser.add_argument('--Image_path' , default='images_10000.npy',type=str,help='Real Image path')
    parser.add_argument('--W_file' , default='W_10000.npy',type=str,help='W file name')
    parser.add_argument('--Semantic_file' , default='semantic_top_32',type=str,help='Semantic File name')
    parser.add_argument('--S_file' , default="SFlat_10000.npy" , type = str , help= "SFlat file name")
    parser.add_argument('--Attribute_file' , default="attribute_10000.npy" , type = str , help= "attribute file name")

    opt = parser.parse_args()

    base_path = opt.base_path
    w_file = opt.W_file
    image_path = opt.Image_path
    semantic_file = opt.Semantic_file
    s_file = opt.S_file
    attribute_file = opt.Attribute_file

    gpu_count = torch.cuda.device_count()

    if opt.mode == "Manipulate":
        run(Manipulate, cfg, description='StyleGAN', default_config='configs/ffhq.yaml',world_size=gpu_count)
    elif opt.mode == "ManipulateReal":
        run(ManipulateReal, cfg, description='StyleGAN', default_config='configs/ffhq.yaml',world_size=gpu_count)
    elif opt.mode == "AttributeDependent":
        run(AttributeDependent, cfg, description='StyleGAN', default_config='configs/ffhq.yaml',world_size=gpu_count)
    elif opt.mode == "PercentageLocalized":
        run(PercentageLocalized, cfg, description='StyleGAN', default_config='configs/ffhq.yaml',world_size=gpu_count)
        