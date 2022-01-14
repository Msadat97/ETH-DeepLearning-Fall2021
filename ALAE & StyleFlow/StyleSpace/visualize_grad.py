#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:51:06 2020

@author: wuzongze
"""
import os
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import time
import argparse
from tqdm import tqdm


def ExpendSMask(semantic_masks,num_semantic):
    
    semantic_masks2=[]
    for i in range(num_semantic):
        tmp=semantic_masks==i
        semantic_masks2.append(tmp)
    semantic_masks2=np.array(semantic_masks2)
    semantic_masks2=np.transpose(semantic_masks2, [1,0,2,3])
    return semantic_masks2
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict pose of object')
    
    parser.add_argument('-gradient_folder',default='./npy/ffhq/gradient_mask_32',type=str,help='path to gradient_mask_32') 
    parser.add_argument('-semantic_path',default='./npy/ffhq/semantic_mask.npy',type=str,help='path to semantic_mask') 
    parser.add_argument('-save_folder',default='./npy/ffhq/align_mask_32',type=str,help='path to save folder') 
    
    parser.add_argument('-img_sindex',default='0',type=str,help='path to model file') 
    parser.add_argument('-num_per',default='4',type=str,help='path to model file') 
    
    opt = parser.parse_args()
    
    #%%
    out_size=32
    
    tmp=os.path.join(opt.gradient_folder,opt.img_sindex)
    with open(tmp, 'rb') as handle:
        var_grad = pickle.load(handle)
        semantic_masks=np.load(opt.semantic_path) 
    
    num_semantic=int(semantic_masks.max()+1)
    semantic_masks=semantic_masks[int(opt.img_sindex):(int(opt.img_sindex)+int(opt.num_per))]
    semantic_masks2=ExpendSMask(semantic_masks,num_semantic)
    
    mask_size=semantic_masks2.shape[-1]
    step=int(mask_size/out_size)
    
    semantic_masks2=semantic_masks2.reshape(int(opt.num_per),num_semantic,out_size,step,out_size,step)
    
    semantic_masks2=np.sum(semantic_masks2,axis=(3,5))
    semantic_masks2_single=np.argmax(semantic_masks2,axis=1)
    
    semantic_masks2=ExpendSMask(semantic_masks2_single,num_semantic)

    #%%
    all_scores=[]
    print("Total number of layers: ",len(var_grad))
    for linex in range(5):
        print('layer index: ',linex)
        layer_g=var_grad[linex]
        num_img,num_channel,_=layer_g.shape
        for img_index in tqdm(range(2)):
#            print(linex,img_index)
            for cindex in range(1):
                mask=layer_g[img_index,cindex].reshape((3,out_size,out_size))
                mask2=np.abs(mask).mean(axis=0).numpy()  #need code 
                plt.imshow(mask2)
                plt.show()