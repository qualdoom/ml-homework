import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # pip install Pillow
from functions_vectorized import convert_image

img = np.asarray(Image.open('image.jpg').convert('RGB'))

data = convert_image(img, np.array([0.299, 0.587, 0.114]))

print(data)

pudge = Image.fromarray(data)

if pudge.mode != 'RGB':
    pudge = pudge.convert('RGB')

pudge.save('lol.jpg')
plt.imshow(pudge)
