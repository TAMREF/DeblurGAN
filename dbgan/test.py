import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import numpy as np
from PIL import Image
import click

from model import generator_model
from utils import load_images, deprocess_image


def test(batch_size,now,gen_num):
    data = load_images('../images/test', batch_size)
    y_test, x_test = data['B'], data['A']
    g = generator_model()
    print('./weights/{}/generator_{}.h5'.format(now,gen_num))
    g.load_weights('./weights/{}/generator_{}.h5'.format(now,gen_num))
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)

    for i in range(generated_images.shape[0]):
        print('{}/{}'.format(i+1,generated_images.shape[0]))
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output = np.concatenate((y, x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save('../images/test_result/results{}.JPEG'.format(i),'JPEG')


@click.command()
@click.option('--batch_size', default=30, help='Number of images to process')
@click.option('--now', default='79', help='Month,Date for training')
@click.option('--gen_num',default=0, help='epoch num of desired data')
def test_command(batch_size,now,gen_num):
    return test(batch_size,now,gen_num)


if __name__ == "__main__":
    test_command()
