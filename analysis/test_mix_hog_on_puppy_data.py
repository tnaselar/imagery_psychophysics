import numpy as np
from PIL import Image
from object_parsing.src.image_objects import view_mask
from imagery_psychophysics.src.mixture_of_histograms import mix_hog, view_lkhd_dist_as_images, view_MAP_as_image
from matplotlib.pyplot import plot, imshow, show

log_file = open('/musc.repo/Data/katie/testpics4_data/katie_2014_Feb_26_1450.log', 'r')
log_data = log_file.readlines()

probe_images_path = '/musc.repo/Data/katie/testpics4_probes/cloud_pics_003054/'

probe_image_size = (500,500)

V = np.prod(probe_image_size)

ovp = 29 #value of the "ON" pixels in the probe images


##get stimuli
stim_string = 'Set image_2 image=/musc.repo/Data/katie/testpics4_probes/cloud_pics_003054/cloud_imsize-500_cloud-100_1204.png'
probe_image_file = []
for sublist in log_data:
  if stim_string in sublist:
   probe_image_file += [sublist[sublist.find('image=')+63:]]
    

##get corresponding keypresses
keyp_string = 'Keypress:'
keypress = []
for sublist in log_data:
  if keyp_string in sublist:
    sub_string = sublist[sublist.find('Keypress')+10]
    if sub_string=='n' or sub_string=='y':
      keypress += sub_string

##get the "yeses"
yeses = [ii for ii,ss in enumerate(keypress) if ss=='y']
M = len(yeses)

##read-in all the image
probes = np.zeros((V,M))
for ii,yy in enumerate(yeses):
  foo = np.array(Image.open(probe_images_path+probe_image_file[yy]).convert('L').resize(probe_image_size))&ovp/ovp
  probes[:,ii] = foo.reshape(V)
  

##view the sum of all probes
imshow(probes.sum(axis=1).reshape(probe_image_size))

##initialize a model
K = 7
##prior
naive_prior_dist = np.random.dirichlet(np.ones(K)).reshape((1,K))
naive_lkhd_dist = np.random.dirichlet(np.ones(V), size=K).T

##create new model
mh = mix_hog(naive_prior_dist, naive_lkhd_dist)

##EM
mh.smoothing_param = .05
mh.run_em(probes)

##view likelihood dist
view_lkhd_dist_as_images(mh.lkhd_dist)

##view MAP
eye_probe = np.eye(V)
post_dist= mh.posterior(eye_probe)
view_MAP_as_image(post_dist)

