from PIL import Image
from numpy import random
N = 125000

##open an image
im = Image.open('/Data/Pictures/google/Andreas+Gursky/png/googleimage426.png')

##create a random image
rand_matrix = random.randint(0, 255, (500, 500, 3)).astype('uint8')
rand_image = Image.fromarray(rand_matrix)

##select N random pixels
rand_pix_coors = random.randint(0,499, (N,2))

##replace the random values at these pixels with the original values
for ii in rand_pix_coors:
  coor = tuple(map(int, ii))
  rand_image.putpixel(coor, im.getpixel(coor))
