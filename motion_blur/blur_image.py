import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import misc
from scipy.signal import signaltools
from generate_PSF import PSF
from generate_trajectory import Trajectory
from random import randint
import sys

#Astropy library to handle Maxim DL FITS file
#from astropy.io import fits
#rawpy library to handle NEF file
import rawpy
from PIL import Image

DEBUG = True
tam_print = print if DEBUG else None

class BlurImage(object):

	def __init__(self, image_path, PSFs=None, part=None, path__to_save=None):
		"""

		:param image_path: path to square, RGB image.
		:param PSFs: array of Kernels.
		:param part: int number of kernel to use.
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
				blured[:, :, 0] = np.array(signaltools.fftconvolve(blured[:, :, 0], tmp, mode='full'))
				blured[:, :, 1] = np.array(signaltools.fftconvolve(blured[:, :, 1], tmp, mode='full'))
				blured[:, :, 2] = np.array(signaltools.fftconvolve(blured[:, :, 2], tmp, mode='full'))
				blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
				blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
				result.append(np.abs(blured))
		else:
			psf = psf[0]
			tmp = np.pad(psf, delta // 2, 'constant')
			tam_print(tmp.size)
			cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
								   dtype=cv2.CV_32F)
			blured[:, :, 0] = np.array(signaltools.fftconvolve(blured[:, :, 0], tmp, mode='same'))
			blured[:, :, 1] = np.array(signaltools.fftconvolve(blured[:, :, 1], tmp, mode='same'))
			blured[:, :, 2] = np.array(signaltools.fftconvolve(blured[:, :, 2], tmp, mode='same'))
			blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
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
				cv2.imwrite(path_string, self.result[0] * 65535)
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


if __name__ == '__main__':
	folder = '../images/astro_data/Lights'
	folder_to_rgb = '../images/astro_rgb'
	folder_to_save = '../images/astro_img_blurred'
	params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
	num_per_img = 10

	#NEF to PIL image
	if len(os.listdir(folder_to_rgb)) == 0:

		sq_size = 2868

		for filename in os.listdir(folder):
			if not '.NEF' in filename: continue
			raw_filename = filename.split('.')[0]
			tam_print(raw_filename)
			raw = rawpy.imread(os.path.join(folder, filename))
			rgb = raw.postprocess()
			img = Image.fromarray(rgb)  # Pillow image
			cut_size = min(sq_size, img.size[0], img.size[1])
			spx = cut_size // 3
			epx = spx * 2
			for i in range(num_per_img):
				k1 = randint(0,spx)
				k2 = randint(0,spx)
				img_cropped = img.crop((spx+k1,spx+k2,epx+k1,epx+k2))
				for j in range(4):
					img_rot = img_cropped
					if j == 1:
						img_rot = img_cropped.transpose(Image.ROTATE_90)
					elif j == 2:
						img_rot = img_cropped.transpose(Image.ROTATE_180)
					elif j == 3:
						img_rot = img_cropped.transpose(Image.ROTATE_270)
					for k in range(2):
						if k > 0:
							img_rot = img_rot.transpose(Image.FLIP_TOP_BOTTOM)
						save_filename = raw_filename + '_' + str(i) + str(j) + str(k)
						img_rot.save(os.path.join(folder_to_rgb, save_filename)+'.TIFF','TIFF')
		tam_print('TIFF conversion finished')
	else:
		tam_print('TIFF conversion was already finished')
	for path in os.listdir(folder_to_rgb):
		tam_print(path)
		trajectory = Trajectory(canvas=64, max_len=60, expl=np.random.choice(params)).fit()
		psf = PSF(canvas=64, trajectory=trajectory).fit()
		BlurImage(os.path.join(folder_to_rgb, path), PSFs=psf,
				  path__to_save=folder_to_save, part=np.random.choice([1, 2, 3])).\
			blur_image(save=True,show=False)
