# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Represent an object map using a simple mixture-of-multinomials model

# <headingcell level=4>

# create environment

# <codecell>

#create environment
import numpy as np
from PIL import Image
from object_parsing.src.image_objects import view_mask
from imagery_psychophysics.src.mixture_of_histograms import mix_hog, view_lkhd_dist_as_images
from matplotlib.pyplot import plot, imshow, show
# <headingcell level=4>

# grab an image / object map

# <codecell>

cur_image = 3600
image_path = '/musc.repo/Data/shared/my_labeled_images/pictures/%0.6d.png'
mask_path = '/musc.repo/Data/shared/my_labeled_images/labeled_image_maps/%0.6d.png'
img = Image.open(image_path %(cur_image))
msk = Image.open(mask_path %(cur_image))
msk=msk.resize(size=(64,64))
img.show()
view_mask(msk)
mask_array = np.array(msk)
N = mask_array.shape
V = np.prod(N)
print 'vocabulary size: %d' %(V)

# <headingcell level=4>

# represent the image mask as the parameters of a mixture-of-histograms generative model

# <codecell>

#K = number of objects
K = len(np.unique(mask_array))

##to create prior, count the number of pixels in each document
L = [sum(sum(mask_array==ii)).astype('double') for ii in np.arange(K)+1]
print 'total number of pixels in each document:'
print L
print

prior_dist = np.array(L/V).reshape((1,K)) ##fraction of pixels in each object
print 'object prior:' 
print prior_dist

##to create the likelihood distributions, treat each object mask as the parameters of a multinomial
def format_mask_as_lkhd():
    return np.hstack([np.atleast_2d(np.array(mask_array==ii).flatten()).T for ii in np.arange(K)+1])

lkhd_dist = format_mask_as_lkhd()/L
view_lkhd_dist_as_images(lkhd_dist)

##initialize a mix_hog model
mh = mix_hog(prior_dist, lkhd_dist)
mh.print_dimensions()

lkhd_dist.sum(axis=0)

# <headingcell level=4>

# sample multiple "probes" from the model

# <codecell>

##number of documents / probes
M = 1000

##number of pixels per document
P = 50

probes = np.zeros((V,M))

##use prior to determine number of probes per object
probes_per_object = np.random.multinomial(M, np.squeeze(mh.prior_dist))

##generate object patches
cnt = 0
for ii,nn in enumerate(probes_per_object):
    for jj in range(nn):
        dx = np.random.choice(np.arange(V), size=P, replace=False, p=np.atleast_1d(mh.lkhd_dist[:,ii]))
        probes[dx, cnt] = 1
        cnt += 1

##check a few probes
imshow(probes[:,999].reshape(N))
show()
# <headingcell level=4>

# re-estimate parameters of model

# <codecell>

print probes.shape

##EM
mh.smoothing_param = 10
mh.run_em(probes)




