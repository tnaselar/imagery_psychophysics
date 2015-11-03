
# coding: utf-8

#### establish enivornment

# In[1]:

##establish multi-engine client
from scoop import futures


# In[2]:

import numpy as np
from imagery_psychophysics.src.model_z import model_z
from imagery_psychophysics.src.probes import probes
from imagery_psychophysics.src.stirling_maps import sparse_point_maps
from imagery_psychophysics.src.object_map import object_map as om
from imagery_psychophysics.src.poisson_binomial import poisson_beta_binomial as pbb
from PIL.Image import fromarray
import time
#%load_ext autoreload
#%autoreload 2


# In[3]:

##local only
#%matplotlib inline
#import matplotlib.pyplot as plt


#### distribute key parameters

# In[4]:

K = 4 ##the number of clusters in what will be the target map
on_off_prob = (.85,.15)
seed_point_rows = 3
seed_point_cols = 3
size_of_field = 32


#### create probes

# In[5]:

probe_set = probes(size_of_field=size_of_field)
probe_set.masks = probe_set.masks.reshape((probe_set.num_probes, size_of_field, size_of_field))


#### generate a simulated target map

# In[6]:

spm = sparse_point_maps(seed_point_rows, seed_point_cols, size_of_field, cluster_pref='all', number_of_clusters = K)
spm.scatter()
target_map = spm.nn_interpolation(dx=[1000])
target_image = fromarray(np.squeeze(target_map).astype('uint8'))
target_om = om(target_image)


# In[7]:

##sanity check: remove all of the duplicates and count
from imagery_psychophysics.src.stirling_maps import stirling_num_of_2nd_kind as snk
import itertools
k = spm.grouping
k.sort()
print 'length of list: %d' %(len(list(k for k,_ in itertools.groupby(k))))
print 'official count: %d' %(snk(seed_point_cols*seed_point_rows, K))


# In[8]:

##see it
num_samples = target_om.count_myself()*20
good_samples = target_om.good_samples(num_samples)
#plt.imshow(target_om.map_as_array)
#plt.scatter(good_samples[:,0], good_samples[:,1],c='k')


#### generate noisy responses to target map

# In[9]:

lkhd = pbb(K, on_off_prob = on_off_prob)
target_noisy_resp = lkhd.sample(target_om.overlap(probe_set.masks))


# In[10]:

##see the noisy responses
#resp_shape = (np.sqrt(probe_set.num_probes), np.sqrt(probe_set.num_probes))
#plt.imshow(target_noisy_resp.reshape(resp_shape).T)
#plt.colorbar()


# In[11]:

##and the likelihood is:
lkhd.sum_log_likelihood(target_noisy_resp, target_om.overlap(probe_set.masks))


# In[ ]:

mental_image = model_z(K, probe_set.masks, target_noisy_resp, size_of_field=size_of_field, cluster_pref = 10)


#### construct a model_z

# In[12]:

def main():
    start = time.time()
    ovl = futures.map(mental_image.overlap, spm.nn_interpolation)
    print 'seconds: %f' %(time.time()-start)
    return ovl


# In[13]:

if __name__ == '__main__':
    mental_image = main()
    from scipy.stats import pearsonr 
    probe_set_shape = ([np.sqrt(probe_set.masks.shape[0])]*2)

    ##predicted response to best stirling map chosen during model init.
    pred_resp = mental_image.predict_response(probe_set.masks)
    r,p = pearsonr(mental_image.resp, pred_resp)
    print r
    print p
    #plt.subplot(1,3,1, title='predicted responses')
    #plt.imshow(pred_resp.reshape((probe_set_shape)))
    #plt.axis('off')
    
    ##optimal predicted responses from the target map used to produce the fake noisy data
    ideal_resp = pbb(K, on_off_prob = (1,0)).sample(target_om.overlap(probe_set.masks))
    r,p = pearsonr(mental_image.resp, ideal_resp)
    print r
    print p
    #plt.subplot(1,3,2, title='ideal responses')
    #plt.imshow(ideal_resp.reshape((probe_set_shape)))
    #plt.axis('off')
    
    ##actual fake responses
    #plt.subplot(1,3,3, title='actual responses')
    #plt.imshow(mental_image.resp.reshape((probe_set_shape)))
    #plt.axis('off')
    
    print mental_image.noise_model.on_prob,mental_image.noise_model.off_prob
    print mental_image.noise_model.alpha,mental_image.noise_model.beta
    
    print mental_image.sum_log_lkhd(target_map)
    print mental_image.sum_log_lkhd(mental_image.best_om)


# In[14]:

#plt.subplot(1,2,1, title = 'best guess')
#plt.imshow(mental_image.best_om)
#plt.axis('off')
#plt.subplot(1,2,2, title = 'target')
#plt.imshow(target_image)
#plt.axis('off')


# In[15]:

##test prediction performance


# In[16]:

##see the selected noise params


# In[17]:

##this is effectively the card trick because we are comparing the likelihood of target map
##to the likelihood of the best regular-grid map


# In[ ]:



