##test some ideas for the object map prior
from PIL import Image
from PIL import ImageDraw
from PIL import ImageChops
from object_parsing.src.image_objects import make_a_blank
from object_parsing.src.image_objects import mask_quantize
import numpy as np
from pylab import *

resolutions = linspace(20, 100, 6).astype('int16')  ##<<Needs to be integers
offsets = linspace(-100,100, 20).astype('int16')
off_x, off_y = np.meshgrid(offsets, offsets)
off_x = off_x.ravel()
off_y = off_y.ravel()
index_num = 127
scale_num = 1
image_size = 500
sample_grid = np.array(make_a_blank(image_size, 0, 'L'))
sample_grid[::resolutions[0], ::resolutions[0]] = 3

##evaluating object completion probability
  ##create fake base object
img = make_a_blank(image_size, 0, 'L')
draw = ImageDraw.Draw(img)
draw.ellipse([(250,250), (310,320)], fill=index_num)
draw.ellipse([(200,200), (300,300)], fill=index_num)
foo = mask_quantize(img, index_num,scale=scale_num)
base_object = np.array(foo)

##create translations of a fake simple "leaf"
img = make_a_blank(500, 0, 'L')
draw = ImageDraw.Draw(img)
draw.ellipse([(200,200), (320,320)], fill=index_num)
leaf_list = [np.array(mask_quantize(img, index_num,scale=scale_num))]
for ii in range(len(off_x)):
  foo = ImageChops.offset(img, off_x[ii], off_y[ii])
  foo = mask_quantize(foo, index_num,scale=scale_num)
  leaf_list.append(np.array(foo))
 

##show what you're working with

imshow(leaf_list[1]+leaf_list[0]+leaf_list[2]+leaf_list[3]+base_object+leaf_list[4])

##calculate leaf indicator over translations of the leaf
def leaf_indicator(base_object, leaf):
  return np.array(base_object==leaf).all()
  
##take average over translations to determine inclusion probability
inclusion_probability = [0]*len(resolutions)
cnt = 0
for stride in resolutions:
  for ii in range(len(leaf_list)):
    inclusion_probability[cnt] += leaf_indicator(base_object[::stride, ::stride], leaf_list[ii][::stride, ::stride])
  cnt += 1
  ##how does amount of subsampling affect inclusion probability?
    ##lower the resolution, the more probable any one base object becomes
  
 
  ##calculate inclusion probability for Q0
  ##include an inclusion probability "fudge" factor. how does it affect probility of object?
  
  ##grab a real object 
  ##grab a group of real leaves, including real object
  ##repeat the above...
  
  ##are there any weird cases? can we see some affect of object geometry?
  
  ##create a "shattered" non-natural object--perhaps by shuffling the image
  ##compare inclusion probability to a "real" object--does real object have higher inclusion probs?
  
 ##recursize calculation of p(Z)
  ##objects will become more and more probable at each level of recursion...that's why empty set is deterministic...
 
 
 
 