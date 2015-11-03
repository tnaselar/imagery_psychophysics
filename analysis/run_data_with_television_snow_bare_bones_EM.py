# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Testing EM for the television snow model

# <headingcell level=3>

# Establish environment

# <codecell>

import numpy as np
import pymc as pm
from imagery_psychophysics.src.probes import probes
from imagery_psychophysics.src.stirling_maps import sparse_point_maps
from imagery_psychophysics.src.stirling_maps import stirling_num_of_2nd_kind as snk
from imagery_psychophysics.src.stirling_maps import Point
from imagery_psychophysics.src.model_z import noise_grid
from matplotlib import pyplot as plt
import itertools
import operator
import pandas as pd
from scipy.misc import comb as nCk
from scipy.misc import imresize
from scipy.stats import pearsonr
from scipy.interpolate import griddata

from deap import base, creator, tools, algorithms
from scoop import futures
import random
from PIL import Image
from scipy import ndimage
from PIL import ImageFilter
from mpl_toolkits.axes_grid1 import ImageGrid

%matplotlib inline

# <headingcell level=3>

# All simulation parameters

# <codecell>

##==critical
size_of_field = 32
n_pix = size_of_field**2
K = 6 ##max of max number of objects
probe_size = 500
init_probs = (.85, .1)  ##<<reasonable start params


##==incidental
seed_point_rows = 3
seed_point_cols = 3
example_probe = 20
num_synthetic_maps_for_prior = 1000

# <headingcell level=3>

# Read and pandify the data

# <codecell>

##the experimental data
probe_exp = pd.load('/musc.repo/scratch/data.pkl')

# <codecell>

##the images -- replace names the pandas dataframe

image_2_path = '/musc.repo/Data/shared/my_labeled_images/pictures/003662.png'
image_2 = Image.open(image_2_path)
image_1_path = '/musc.repo/Data/shared/my_labeled_images/pictures/001676.png'
image_1 = Image.open(image_1_path)
probe_exp.replace(to_replace = 'finalprobeset-1', value = image_1_path, regex=True, inplace=True)
probe_exp.replace(to_replace = 'finalprobeset-2', value = image_2_path, regex=True, inplace=True)
print probe_exp.head()
print probe_exp.tail()

# <codecell>

##the probes: map from image name to set of probes
probe_dict = {image_1_path: '/musc.repo/Data/katie/nbd_probes/finalprobeset-1/just-probes/', \
              image_2_path: '/musc.repo/Data/katie/nbd_probes/finalprobeset-2/just-probes/'}


##processing the probes: accepts a PIL image
def quantreduce(im, sz):
    im = im.filter(ImageFilter.MaxFilter(5))
    r,_,_ = im.resize((sz,sz), Image.BICUBIC).split()
    threshold = 200  
    return np.array(r.point(lambda p: p > threshold and 1))


#get center pixel of probe: input = np.array
def probecenter(im):
    pi, pj = ndimage.measurements.center_of_mass(np.array(im))
    return np.round(pi).astype('int'), np.round(pj).astype('int')
  
##for scaling the probe centers to match the downsampled probes
def rescale(probe_size, size_of_field, points):
    rs = (float(size_of_field)/probe_size)
    return points[:,0]*rs, points[:,1]*rs
    
    

# <codecell>


##extract probe centers from full-size probes
probe_centers = np.array([(probecenter(quantreduce(Image.open(probe_dict[im]+'probe-%d.jpeg' %(pr)),probe_size)))  for im,pr in zip(probe_exp.image, probe_exp.probe)])

##put the reduced-size probes into a list
probe_list = [quantreduce(Image.open(probe_dict[im]+'probe-%d.jpeg' %(pr)),size_of_field) for im,pr in zip(probe_exp.image, probe_exp.probe)]

# <codecell>

##add new columns to data frame containing probe masks and centers for associated with each response...
probe_exp['probe_mask'] = probe_list
probe_exp['probe_center_i'] = probe_centers[:,0]
probe_exp['probe_center_j'] = probe_centers[:,1]
probe_exp.head()

# <codecell>

##check that probes for each rep are the same. enter a few numbers below to checks
foo = probe_exp.groupby(['subj', 'state', 'image', 'probe'])
check_this_one = 256
dx = foo.groups.keys()[check_this_one]
probe_exp.iloc[foo.groups[dx]]

# <codecell>

##grab the centers and sizes of each nbd

 
##get the nbd for plotting the maps
cnt = 0
probe_agg = probe_exp.groupby('image')
fig = plt.figure(1, (10,10))
nbd_center = dict()
nbd_size = dict()
for name,grp in probe_agg:
    nbdpic = grp.probe_mask.sum()>0
    
#     probepic = np.zeros((probe_size, probe_size))
    plt.subplot(1,2,cnt)
    plt.title(name, fontsize=8)
    plt.imshow(nbdpic, interpolation='none', cmap = 'gray')
    nbd_center[name]  = probecenter(nbdpic)
    ii,jj = rescale(probe_size, size_of_field, np.array([grp.probe_center_i, grp.probe_center_j]).T)
    plt.scatter(jj,ii)
    plt.scatter(nbd_center[name][1],nbd_center[name][0],c='r')
    nbd_size[name] = ii.max() - ii.min()
    cnt += 1

# cnt = 0
# for name,grp in probe_agg:
#     plt.subplot(1,2,cnt)
    
#     #plt.scatter(nbd_center[cnt][0],nbd_center[cnt][1],c='r')
#     cnt += 1

# <codecell>

nbd_size

# <codecell>

np.array([grp.probe_center_i, grp.probe_center_j]).T.shape

# <headingcell level=3>

# View the average response maps

# <codecell>

##view average response maps 
response_map_dict = {}
agg = probe_exp.groupby(['subj', 'image', 'state'])
pi,pj = np.meshgrid(range(np.min(probe_centers),np.max(probe_centers),1),range(np.min(probe_centers),np.max(probe_centers),1))
ix = np.zeros((pi.size,2))
ix[:,0] = pi.ravel()
ix[:,1] = pj.ravel()
for key,grp in agg:
    resp = grp.groupby('probe').response.mean()
    pc = np.zeros((resp.size,2))
    pc[:,0] = grp.groupby('probe').probe_center_i.mean() ##<<this just collapses the reps
    pc[:,1] = grp.groupby('probe').probe_center_j.mean()
    response_map = np.zeros((probe_size, probe_size))
    iresponse_map = griddata(pc, resp, ix, method='nearest') ##upsample the response map
    for ii,jj,rr in zip(ix[:,0],ix[:,1],iresponse_map): ##re-format to construct a proper image
        response_map[ii,jj] = rr
    response_map[response_map<1] = None
    response_map_dict[key] = response_map        
            

# <codecell>

#best colormap so far
cmap2 = plt.cm.get_cmap('gist_heat', K) 

# <codecell>

##view: construct an image grid
fig = plt.figure(1, (15., 15.))
grid = ImageGrid(fig, 111, # similar to subplot(111)
                nrows_ncols = (3, 2), # creates 2x2 grid of axes
                axes_pad=0.5, # pad between axes in inch.
                cbar_mode = 'each',
                cbar_pad = .05
                )

##order the response maps for eacse of comparison
cnt = 0
ordered_keys = sorted(sorted(response_map_dict.keys(),key=lambda x: x[1]), key = lambda x: x[0])[:-1] ##<<skip last, no data

for key in ordered_keys:
    img = key[1]
    rmap = response_map_dict[key]
    grid[cnt].imshow(Image.open(img))
    im = grid[cnt].imshow(rmap, interpolation='nearest', alpha = .5, cmap=cmap2, clim=[1,K+1])
    grid[cnt].set_title((key[0], key[2]))
    grid[cnt].set_xticks([])
    grid[cnt].set_yticks([])
    #grid.cbar_axes[cnt].colorbar(im,ca)
    
    grid[cnt].cax.colorbar(im)
    grid[cnt].cax.set_yticks(range(0,K+1,1))
    #plt.colorbar(im, ax=grid[cnt])
    #plt.axis('off')
    cnt += 1
    

# <codecell>

probe_exp.groupby(['subj', 'image', 'state']).response.max()

# <headingcell level=3>

# Some functions for manipulating object maps

# <codecell>

##convert from membership representation to binarized object map ("bom") represenation
def mem_2_bom(M, num_pix,max_number_of_objects):
        Z = np.zeros((num_pix, max_number_of_objects))
        for ii in np.unique(M)-1: ##<<assume that min_value = 1
            Z[np.nonzero(M==(ii+1))[0],ii] = 1
        return Z
    
def bom_2_mem(M):
    new_obj = np.zeros(M.shape[0],dtype='int')
    for kk in range(M.shape[1]):
        new_obj[M[:,kk].astype('bool')] = kk+1
    return new_obj

##outer product of an object map
def outer_map(M,num_pix,max_number_of_objects):
        Z = mem_2_bom(M, num_pix,max_number_of_objects)
        return Z.dot(Z.T)

##another helpful data conversions
def intlist_to_bitlist(number_of_objects,intlist):
  '''
  intlist_to_bitlist(number_of_objects,intlist)
  takes a list of integers and converts to a binary code
  if number_of_objects = 10, and intlist = [1,2,3,7] then returns
  numpy array [0 1 1 1 0 0 0 1 0 0]
  '''
  x = np.zeros(number_of_objects)
  x[intlist] = 1
  return x    
    
##overlap between a probe and the objects in a map
def overlap(probes, one_map):
  '''
  overlap(probes, map)
  probes ~ N x D x D binary masks. 
  map ~ D x D object map.
  applies the probes to one object map
  returns a  number_of_probes x number_of_objects array
  each column is a bit string showing which of the objects a probe overlaps
  '''
  number_of_objects = len(np.unique(one_map))
  ##note the -1 below: the objects in the maps are assumed to have non-zero indices
  return np.array([intlist_to_bitlist(number_of_objects,np.nonzero(np.unique(q))[0]-1) for q in one_map*probes])

def response(ov):
    return np.sum(ov, axis=1)

def pair_objects(m1, m2):
    n_pix = m1.size
    n_obj_1 = np.unique(m1).size
    n_obj_2 = np.unique(m2).size
    m1_bom = mem_2_bom(m1, n_pix, n_obj_1)
    m2_bom = mem_2_bom(m2, n_pix, n_obj_2)
    pr = np.zeros((n_obj_1, n_obj_2))
    for kk in range(n_obj_1):
        for jj in range(n_obj_2):
            pr[kk,jj] = pearsonr(m1_bom[:,kk], m2_bom[:,jj])[0]
    if n_obj_1 > n_obj_2:
        return bom_2_mem(m1_bom[:,np.argmax(pr,0)]), m2
    else:
        return bom_2_mem(m2_bom[:,np.argmax(pr,1)]), m1
        

# <headingcell level=3>

# an object map prior

# <codecell>

##here we estimate the correlation stucture of a generic object map using
##many synthetically generate maps

Lambda = np.zeros((n_pix,n_pix))

for _ in range(num_synthetic_maps_for_prior):
    k = np.random.randint(1,K+1)
    rand_map = sparse_point_maps(seed_point_rows, seed_point_cols, size_of_field, cluster_pref='random', number_of_clusters = k)
    rand_map.scatter()
    rand_map = np.squeeze(rand_map.nn_interpolation()).reshape((n_pix,1))
    Lambda += outer_map(rand_map,n_pix,K)
Lambda /= num_synthetic_maps_for_prior

def prior(value, n_pix = n_pix, max_value = K, hyper_params = Lambda[np.triu_indices(n_pix)]):
    """2nd order prior for object maps"""
    #num_pairs = hyper_params.size
    if (np.min(value) < 1) or (np.max(value) > max_value):
        return -np.Inf
    else:
        on_offs = outer_map(value, n_pix, max_value)
        on_offs = on_offs[np.triu_indices(n_pix)].astype('int')
    
    return pm.bernoulli_like(on_offs, hyper_params.ravel())

# <codecell>

fig = plt.figure(2, (2.5,2.5))
plt.imshow(Lambda, cmap = 'bone',vmin = 0, vmax=1)
plt.title('generic prior\nfor object maps')
plt.axis('off')
plt.colorbar()

# <headingcell level=3>

# the likelihood function

# <codecell>

##lkhd function

def counts(r,d,n):
    return np.array([nCk(d,m)*nCk(n-d, r-m) for m in range(min(r,d)+1)])

def lkhd(r,d,n,p_on,p_off):
    probs = np.array([(1-p_on)**(d-m) * (p_on)**m * (p_off)**(r-m) * (1-p_off)**(n-d-r+m) for m in range(min(r,d)+1)])
    return counts(r,d,n).dot(probs)

def multi_lkhd(r,d,n,p_on,p_off):
    return np.sum(np.array(map(lambda x,y: np.log(lkhd(x,y,n,p_on,p_off)), r,d)))



# <headingcell level=3>

# A procedure for MAP estimate of object map using grid search over stirling maps 

# <codecell>

##make the maps to do a grid search over
def make_maps(size_of_nbd, nbd_center, num_clusters):
    nbdx,nbdy = nbd_center
    map_grid = sparse_point_maps(seed_point_rows, seed_point_cols, size_of_nbd, cluster_pref='all', number_of_clusters = num_clusters)
    map_grid.size_of_field = size_of_field
    map_grid.uniform_shift((nbdx-size_of_nbd/2.)/size_of_field, (nbdy-size_of_nbd/2.)/size_of_field)
    return map_grid.nn_interpolation(), map_grid.flatten()

foo,pts = make_maps(nbd_size[nbd_size.keys()[0]], nbd_center[nbd_center.keys()[0]], 4)



# <codecell>

plt.imshow(np.squeeze(foo[7000,:,:]), cmap = 'Pastel1')
plt.scatter(pts[:,0],pts[:,1])
foo.shape

# <codecell>

##for running the grid search
def MAP(init_probs, r, probes, map_grid, n):
    p_on, p_off = init_probs 
    D =  map_grid.shape[0]
    

    likes = np.empty(D, dtype = 'float64')
    for ii,mg in enumerate(map_grid):
        pred_resp = response(overlap(probes,mg)).astype('int')
        try:
            likes[ii] = multi_lkhd(r, pred_resp, n, p_on, p_off)
        except:
            1/0
    
    #print likes
    best_dx = np.argmax(likes)
    cur_best = np.squeeze(map_grid[best_dx,:, :])
    pred_resp = response(overlap(probes,cur_best))
    pred_score = pearsonr(r, pred_resp)
    return best_dx, cur_best, pred_resp, pred_score, likes[best_dx]


# <headingcell level=3>

# a max likelihood procedure for estimate of noise params

# <codecell>

def MXL(r, pred_resp, n):
    ##run a grid search for p_on, p_off
    p_on, p_off = noise_grid(20,20)
    D = len(p_on)
    p_on = np.array(p_on)
    p_off = np.array(p_off)
    likes = np.array([multi_lkhd(r,pred_resp,n, p[0],p[1]) for p in zip(p_on,p_off)])
    best_dx = np.argmax(likes)
    best_p_on = p_on[best_dx]
    best_p_off = p_off[best_dx]
    return best_dx, best_p_on, best_p_off, likes[best_dx]

# <codecell>

type(grp[grp.rep < 1].response.astype('int').iloc[0])

# <headingcell level=3>

# Bare-bones EM (map --> max lkhd --> map --> ..)

# <codecell>

##==training/testing
agg = probe_exp.groupby(['subj', 'image', 'state'])
results_dict = dict()
for name, grp in agg:
    print '=========subject: %s, image: %s, state: %s' %(name)
    train_dx = grp.rep<1
    responses = grp.loc[train_dx].response.astype('int')
    probes = np.zeros((responses.size,size_of_field,size_of_field), dtype='int')
    cnt = 0
    for p in grp.loc[train_dx].probe_mask.values:
        probes[cnt,:,:] = p.astype('int')
        cnt += 1
    old_dx = -1   
    best_lkhd = -np.Inf
    
    n = np.max(responses)
    #toats = np.sum([snk(seed_point_cols*seed_point_rows, ii) for ii in range(1,n+1)])
    map_grid,_ = make_maps(nbd_size[name[1]], nbd_center[name[1]], n)

    ##find best map, assuming reasonable parameters
    print 'finding new map...'
    new_dx, cur_map, pred_resp, cur_score, map_lkhd = MAP(init_probs, responses, probes, map_grid, n)
    print 'current best map (lkhd): %d (%f), previous best map: %d' %(new_dx, map_lkhd, old_dx)
    ##while best map is novel, 
    while new_dx != old_dx:
        old_dx = new_dx
        ##find best noise params, assuming best map
        print 'finding new noise params...'
        _, cur_p_on, cur_p_off, xl = MXL(responses.astype('int'), pred_resp.astype('int'), n)
        print 'cur lkhd: %f, best lkhd: %f' %(xl, best_lkhd)
        print 'cur noise params: %f, %f' %(cur_p_on, cur_p_off)
        if xl > best_lkhd:
            best_lkhd = xl
            best_map = cur_map
            best_score = cur_score
            best_dx = new_dx
            best_p_on = cur_p_on
            best_p_off = cur_p_off
        ##find best map, assuming best noise params
        print 'finding new map...'
        new_dx, cur_map, pred_resp, pred_score, map_lkhd = MAP((cur_p_on,cur_p_off), responses, probes,  map_grid, n)
        print 'current best map (lkhd): %d (%f), previous best map: %d' %(new_dx, map_lkhd, old_dx)
    results_dict[name] = (best_lkhd, best_map, best_score, best_dx, best_p_on, best_p_off)

# <headingcell level=3>

# View results

# <codecell>

##make a dataframe
em_results = pd.DataFrame(results_dict).T
em_results.columns = ['likelihood', 'object_map', 'pred_score', 'map_index', 'p_on', 'p_off']
em_results

##save a dataframe
em_results.to_hdf('/musc.repo/Data/tnaselar/imagery_psychophysics/em_results.h5','bare_bones_em_v2')

# <codecell>

##load a dataframe
#em_results = pd.read_hdf('/musc.repo/Data/tnaselar/imagery_psychophysics/em_results.h5','bare_bones_em_v2')

# <codecell>

em_results

# <codecell>

##what is "n" (the max number of objects) for each experiment?
n_obj = em_results.object_map.apply(lambda x: len(np.unique(x)))
n_obj

# <codecell>

# n_obj.plot(kind = 'bar', stacked='True', title = 'max number of objects')

# <codecell>

##look at image maps
#ocm = plt.cm.get_cmap('', K) 
##view: construct an image grid
view_size = 256

def mycm(num):
    return plt.cm.get_cmap('hot', int(num))

fig = plt.figure(1, (10., 10.))
grid = ImageGrid(fig, 111, # similar to subplot(111)
                nrows_ncols = (3, 2), # creates 2x2 grid of axes
                axes_pad=0.85, # pad between axes in inch.
                cbar_mode = 'none',
                cbar_pad = .05
                )
cnt = 0
for mm,dx in zip(em_results['object_map'],em_results['object_map'].index):
    n = len(np.unique(mm))
    mm = imresize(mm,(view_size,view_size), interp='nearest')
    grid[cnt].imshow(Image.open(dx[1]).resize((view_size, view_size)))
    im = grid[cnt].imshow(mm, interpolation='nearest', alpha = .5, cmap=mycm(n), clim=[1,np.max(mm)])
    total_lkhd = em_results.likelihood.loc[dx]
    prob_on = em_results.p_on.loc[dx]
    prob_off = em_results.p_off.loc[dx]
    pred_score = em_results.pred_score.loc[dx]
    tit = '%s, %s, lkhd: %0.2f\nprobs: (%0.2f, %0.2f)\npred. score: %0.2f' %(dx[0], dx[2], total_lkhd, prob_on, prob_off, pred_score[0])
    grid[cnt].set_title(tit)
    grid[cnt].set_xticks([])
    grid[cnt].set_yticks([])
    grid[cnt].cax.colorbar(im)
    #grid[cnt].cax.set_yticks([])
    plt.axis('off')
    cnt += 1
    
    

# <codecell>

##plot the distributions across "d" for each case: NOTE: NEED TO BE ORDERED CORRECTLY
fig = plt.figure(1, (10,10))
cnt = 0
for dx in em_results.index:
    plt.subplot(3,2,cnt)
    total_lkhd = em_results.likelihood.loc[dx]
    prob_on = em_results.p_on.loc[dx]
    prob_off = em_results.p_off.loc[dx]
    pred_score = em_results.pred_score.loc[dx]
    n = np.unique(em_results.object_map.loc[dx]).size
    tit = '%s, %s, lkhd: %0.2f\nprobs: (%0.2f, %0.2f)\npred. score: %0.2f' %(dx[0], dx[2], total_lkhd, prob_on, prob_off, pred_score[0])
    plt.title(tit)
    rng = range(0,n+1,1)
    for dd in rng:
        evid = np.zeros(len(rng))
        for rr in rng:
            evid[rr] = lkhd(rr,dd,n,prob_on,prob_off)
        plt.plot(rng, evid, 'o-', label='d=%d' %(dd))
    plt.xticks(rng)
    plt.legend(loc = 'upper right', ncol = 7, fontsize=6)
    cnt += 1
plt.tight_layout()

# <codecell>


