from PIL import Image
import os
import numpy as np
di = './trainA'
for name in os.listdir(di):
    img = Image.open(os.path.join(di,name))
    arr = np.array(img)
    print(arr.shape)
