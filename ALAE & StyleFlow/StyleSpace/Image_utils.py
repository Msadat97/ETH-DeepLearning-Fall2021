import numpy as np
from PIL import Image

import argparse
from tqdm import tqdm
def resize(images , new_scale , num_img):

    new_images = []
    ctr  = 0
    for img in tqdm(images):
        img=Image.fromarray(img).resize((new_scale,new_scale),Image.LANCZOS)
        img=np.array(img)
        new_images.append(img)

        ctr += 1
        if ctr == num_img:
            return new_images

    return new_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate the DCI of given latent codes')
    parser.add_argument('-images_path',type=str,help='path to images')
    parser.add_argument('-save_path',type=str,help='path to save file')
    
    parser.add_argument('-mode',default='scale',type=str,choices=['scale']) 
    parser.add_argument('-scale' , default=256 , type=int , help="The scale of the new images")
    parser.add_argument('-num_img' , default=10000 , type=int , help='Number of images to read')

    args = parser.parse_args()
    
    if args.mode=='scale':
        images = np.load(args.images_path)
        scaled_imgs = resize(images , args.scale, args.num_img)
        np.save(args.save_path,scaled_imgs )
        