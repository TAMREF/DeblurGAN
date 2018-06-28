folder = '../images/astro_data/Lights'
folder_to_rgb = '../images/astro_rgb'
folder_to_save = '../images/astro_img_blurred'
params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
# NEF to PIL image
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
        img_cropped = img.crop((spx, spx, epx, epx))
        img_cropped.save(os.path.join(folder_to_rgb, raw_filename) + '.TIFF', 'TIFF')
    tam_print('TIFF conversion finished')
else:
    tam_print('TIFF conversion was already finished')