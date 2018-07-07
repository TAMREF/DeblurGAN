import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import misc
from scipy.signal import signaltools
from generate_PSF import PSF
from generate_trajectory import Trajectory
from random import randint
from PILutils import PIL2array
import sys

#Astropy library to handle Maxim DL FITS file
#from astropy.io import fits
#rawpy library to handle NEF file
import rawpy
from PIL import Image

DEBUG = True
GRAYSCALE_TRS = 1.5
tam_print = print if DEBUG else None

class BlurImage(object):

    def __init__(self, image_path, PSFs=None, part=None, path__to_save=None):
        """

        :param image_path: path to square, RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        """
        if os.path.isfile(image_path):
            self.image_path = image_path
            self.original = misc.imread(self.image_path)
            tam_print('ORIGINAL IMAGE : ',self.original.shape)
            self.shape = self.original.shape
            if len(self.shape) < 3:
                print(self.shape)
                raise Exception('We support only RGB images yet.')
            elif self.shape[0] != self.shape[1]:
                raise Exception('We support only square images yet.')
        else:
            raise Exception('Not correct path to image.')
        self.path_to_save = path__to_save
        if PSFs is None:
            if self.path_to_save is None:
                self.PSFs = PSF(canvas=self.shape[0]).fit()
            else:
                self.PSFs = PSF(canvas=self.shape[0], path_to_save=os.path.join(self.path_to_save,
                                                                                'PSFs.png')).fit(save=True)
        else:
            self.PSFs = PSFs

        self.part = part
        self.result = []

    def blur_image(self, save: object = False, show: object = False) -> object:
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]
        yN, xN, channel = self.shape
        key, kex = self.PSFs[0].shape
        delta = yN - key
        assert delta >= 0, 'resolution of image should be higher than kernel'
        result=[]
        if len(psf) > 1:
            for p in psf:
                tmp = np.pad(p, delta // 2, 'constant')
                tam_print(tmp.size)
                cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # blured = np.zeros(self.shape)
                blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
                blured[:, :, 0] = np.array(signaltools.fftconvolve(blured[:, :, 0], tmp, mode='same'))
                blured[:, :, 1] = np.array(signaltools.fftconvolve(blured[:, :, 1], tmp, mode='same'))
                blured[:, :, 2] = np.array(signaltools.fftconvolve(blured[:, :, 2], tmp, mode='same'))
                blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
                result.append(np.abs(blured))
        else:
            psf = psf[0]
            tmp = np.pad(psf, delta // 2, 'constant')
            tam_print(tmp.size)
            cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_L1, dtype=cv2.CV_32F)
            blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)
            blured[:, :, 0] = np.array(signaltools.fftconvolve(blured[:, :, 0], tmp, mode='same'))
            blured[:, :, 1] = np.array(signaltools.fftconvolve(blured[:, :, 1], tmp, mode='same'))
            blured[:, :, 2] = np.array(signaltools.fftconvolve(blured[:, :, 2], tmp, mode='same'))
            blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
            tam_print('Blurred size : ', blured.shape)
            result.append(np.abs(blured))
        self.result = result
        if show or save:
            self.__plot_canvas(show, save)

    def __plot_canvas(self, show, save):
        tam_print('Got a call!')
        if len(self.result) == 0:
            raise Exception('Please run blur_image() method first.')
        else:
            plt.close()
            plt.axis('off')
            fig, axes = plt.subplots(1, len(self.result), figsize=(10, 10))
            if len(self.result) > 1:
                for i in range(len(self.result)):
                        axes[i].imshow(self.result[i])
            else:
                plt.axis('off')

                plt.imshow(self.result[0])
            if show and save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                path_string = os.path.join(self.path_to_save, self.image_path.split('\\')[-1])
                cv2.imwrite(path_string, self.result[0])
                plt.show()
            elif save:
                #tam_print('PATH TO SAVE : ', self.path_to_save, 'IMAGE PATH : ', self.image_path)
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                path_string = os.path.join(self.path_to_save, self.image_path.split('\\')[-1])
                print('path string : ',path_string)
                misc.imsave(path_string,self.result[0])
            elif show:
                plt.show()

def merge_flat(image_path,result_path):
    if len(os.listdir(result_path)) > 0:
        img = Image.open(os.path.join(result_path, 'MasterBias800_03_09_2013.tif'))
        arr_flat = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])
        return np.tile(arr_flat,(3,1,1)).reshape(2868,4320,3)
    tam_print('merging flats')
    L = []
    for filename in os.listdir(image_path):
        if not '.NEF' in filename: continue
        raw_filename = filename.split('.')[0]
        tam_print(raw_filename)
        raw = rawpy.imread(os.path.join(image_path, filename))
        rgb = raw.postprocess(user_black = 0)
        L.append(rgb)
    mean_flat = np.mean(L,axis=0)
    tam_print(mean_flat.shape)
    tam_print('flat merging process finished')
    img_flat = Image.fromarray(mean_flat.astype(np.uint8))
    img_flat.save(os.path.join(result_path, 'flat_result.JPEG'),'JPEG')
    return mean_flat

if __name__ == '__main__':
    folder = '../images/astro_data/Lights'
    folder_to_rgb = '../images/astro_rgb'
    folder_to_save = '../images/astro_img_blurred'
    folder_flat = '../images/astro_data/Flats'
    folder_flat_result = '../images/astro_data/Flat_results'
    params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
    num_per_img = 10 # TODO : Should be 10
    arr_flat = merge_flat(folder_flat, folder_flat_result)
    #NEF to PIL image
    if True:
    #if len(os.listdir(folder_to_rgb)) == 0:

        sq_size = 2868
        black_thrs = [256,220,230]
        #thrs_mtx = np.repeat(black_thrs,2868*4320).reshape(2868,4320,3)
        for filename in os.listdir(folder):
            if not '.NEF' in filename: continue
            raw_filename = filename.split('.')[0]
            tam_print(raw_filename)
            raw = rawpy.imread(os.path.join(folder, filename))
            rgb = raw.postprocess(user_black = 0)
            print(rgb.dtype)

            #rgb_calib = np.divide(rgb * 256.,  arr_flat)
            #rgb_calib = rgb - arr_flat

            rgb_calib = rgb
            #rgb_calib = np.maximum(rgb,black_thrs)
            #thrs_mtx = np.repeat(black_thrs, rgb_calib.size // 3).reshape(rgb_calib.shape)

            for i in range(3):
                tmp = rgb_calib[:,:,i]
                tmp[tmp < black_thrs[i]] = 0
                rgb_calib[:,:,i] = tmp

            if GRAYSCALE_TRS > 1:
                rgb_normalized = np.mean(rgb_calib,axis=2) * GRAYSCALE_TRS
                for i in range(3):
                    rgb_calib[:,:,i] = rgb_normalized
                print(rgb_calib.shape)
            #rgb_calib = rgb_calib - thrs_mtx
            #rgb_calib = cv2.normalize(rgb_calib, rgb_calib, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            img = Image.fromarray(rgb_calib.astype(np.uint8))  # Pillow image
            cut_size = min(sq_size, img.size[0], img.size[1])
            spx = cut_size // 3
            epx = spx * 2
            for i in range(num_per_img):
                k1 = randint(0,spx)
                k2 = randint(0,spx)
                img_cropped = img.crop((spx+k1,spx+k2,epx+k1,epx+k2))
                for j in range(4): #TODO : Should be 4
                    img_rot = img_cropped
                    if j == 1:
                        img_rot = img_cropped.transpose(Image.ROTATE_90)
                    elif j == 2:
                        img_rot = img_cropped.transpose(Image.ROTATE_180)
                    elif j == 3:
                        img_rot = img_cropped.transpose(Image.ROTATE_270)
                    for k in range(2): #TODO : Should be 2
                        if k > 0:
                            img_rot = img_rot.transpose(Image.FLIP_TOP_BOTTOM)
                        save_filename = raw_filename + '_' + str(i) + str(j) + str(k)
                        img_rot.save(os.path.join(folder_to_rgb, save_filename)+'.JPEG','JPEG')
        tam_print('TIFF conversion finished')
    else:
        tam_print('TIFF conversion was already finished')
    for path in os.listdir(folder_to_rgb):
        tam_print(path)
        if not '.JPEG' in path: continue
        trajectory = Trajectory(canvas=64, max_len=60, expl=np.random.choice(params)).fit()
        psf = PSF(canvas=64, trajectory=trajectory).fit()
        BlurImage(os.path.join(folder_to_rgb, path), PSFs=psf,
                  path__to_save=folder_to_save, part=np.random.choice([1, 2, 3])).\
            blur_image(save=True,show=False)
