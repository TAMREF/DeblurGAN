from PIL import Image
import os
import numpy as np
import rawpy
from PILutils import PIL2array

def rgb_debug(rgb,dbg):
    for filename in os.listdir(rgb):
        if filename == 'dbg': continue
        img = Image.open(os.path.join(rgb,filename))
        raw_name = filename.split('.')[0]
        arr = PIL2array(img)
        print(arr.shape)
        for i in range(3):
            tmp = np.zeros(arr.shape,dtype=np.uint8)
            tmp[:,:,i] = arr[:,:,i]
            img_tmp = Image.fromarray(tmp)
            img_tmp.save(os.path.join(dbg,raw_name)+'RGB'[i]+'.JPEG','JPEG')

def array_debug(path, israw=False):
    for filename in os.listdir(path):
        if israw:
            raw = rawpy.imread(os.path.join(path,filename))
            arr = raw.postprocess()
        else:
            img = Image.open(os.path.join(path,filename))
            arr =  PIL2array(img)
        R = np.mean(arr,axis=(0,1))
        print(R.shape)
        print(filename, ' : ', R)

if __name__ == "__main__":
    #path = '../images/astro_rgb'
    #path = '../images/astro_data/Flats'
    path = '../images/astro_data/Lights'
    path_rgb = '../images/astro_rgb'
    path_dbg = '../images/astro_rgb_dbg'
    path_blr = '../images/astro_img_blurred'
    rgb_debug(path_rgb,path_dbg)
    array_debug(path_dbg)

