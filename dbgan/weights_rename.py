import os
import glob

if __name__ == '__main__':
    folder = '~/Desktop/DeblurGAN/dbgan/weights/79'
    for path in glob.glob(folder):
        print(path)
        path_r = path[:-7]
        os.rename(path,path_r+'.h5')
