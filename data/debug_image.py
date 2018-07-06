from PIL import Image
import os
import numpy as np
import rawpy
from PILutils import PIL2array

if __name__ == "__main__":
    #path = '../images/astro_rgb'
    path = '../images/astro_data/Flats'
    for filename in os.listdir(path):
        '''
        img = Image.open(os.path.join(path,filename))
        arr =  PIL2array(img)
        '''
        raw = rawpy.imread(os.path.join(path,filename))
        arr = raw.postprocess(user_black = 0)

        R = np.mean(arr,axis=(0,1))
        print(R.shape)
        print(filename, ' : ', R)
