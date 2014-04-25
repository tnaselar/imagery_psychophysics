from PIL import Image
from numpy import random

im = Image.open('/Data/Pictures/google/Andreas+Gursky/png/googleimage426.png')

THRESH = 128
SIZE = 500
##make a random image
rand_matrix = random.randint(0, 255, (SIZE, SIZE, 3)).astype('uint8')
rand_pic = Image.fromarray(rand_matrix)

##loop over the rows of the image
for ii in range(SIZE):
  ##loop over the columns
  for jj in range(SIZE):
    ##put the coordinates in a tuple
    coor = tuple(map(int, [ii,jj]))
    ##if all of the R,G and B channels are greater than threshold
    if sum(rand_pic.getpixel(coor)) < 3*THRESH:
      continue
    else: ##replace the pixel color in the random picture with the real picture
      rand_pic.putpixel(coor, im.getpixel(coor)) 

#for val,coor in enumerate(rand_pic.getdata()):
  #if val < THRESH:
    #continue
  #else:
    #im.putpixel(
