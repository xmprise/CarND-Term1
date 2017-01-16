import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('/home/wj-linux/CarND-Term1/Sample_image/test.jpg')
print('This image is: ', type(image), 'with dimensions:', image.shape)

ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)

red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

threshold = (image[:, :, 0] < rgb_threshold[0]) \
          | (image[:, :, 1] < rgb_threshold[1]) \
          | (image[:, :, 2] < rgb_threshold[2])
color_select[threshold] = [0, 0, 0]

plt.imshow(color_select)
mpimg.imsave("test-after.png", color_select)