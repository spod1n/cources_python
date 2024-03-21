import numpy as np
from PIL import Image

if __name__ == '__main__':
    img = Image.open('win_logo_full.jpg')
    img_array = np.array(img)

    white_color = np.array([255, 255, 255])
    black_color = np.array([0, 0, 0])

    mask = np.all(img_array == white_color, axis=-1)
    img_array[mask] = black_color

    img_change = Image.fromarray(img_array)
    img_change.save('win_logo_change.jpg')

    img_change.show()
