##Model "Z", arbitrarily named after the notation in my workbook.
##This class implements model Z, which brings to together an object map, a set of probes, and a set of responses assumed to have
##a poisson_beta_binomial distribution. It includes methods for inferring the object map from the set of probe responses.
##Uses composition instead of inheritance
import numpy as np
from imagery_psychophysics.src.stirling_maps import stirling_num_of_2nd_kind as snk
from imagery_psychophysics.src.probes import probes
#from imagery_psychophysics.src.object_map import dead_leaves as dl
from imagery_psychophysics.src.stirling_maps import sparse_point_maps
#from imagery_psychophysics.src.object_map import object_map as om
#from imagery_psychophysics.src.object_map import binarized_object_map as bom
from imagery_psychophysics.src.poisson_binomial import poisson_beta_binomial as pbb


#from PIL.Image import fromarray

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
  

def noise_grid(noise_grid_x_dns,noise_grid_y_dns ):
  '''
  generate a grid of noise parameters on_prob/off_prob
  noise_grid_x_dns ~ density of noise model param grid, x-axis
  noise_grid_y_dns ~ density of noise model param grid, y-axis
  '''
  small = 10**-3  ##make sure off-prob strictly < on-prob
  dns = noise_grid_x_dns ##density of noise model param grid, x-axis
  D = noise_grid_y_dns ##density of noise model param grid, y-axis
  
  p_on =  [np.array([ii]*np.max([np.ceil(D*ii).astype('int32'),1])) for ii in np.linspace(small,1-small,dns)]
  p_off = [np.linspace(small,ii[0]-small,len(ii)) for ii in p_on]
  p_on = [item for sublist in p_on for item in sublist]
  p_off = [item for sublist in p_off for item in sublist]
  return p_on, p_off
  
  


class model_z(object):
  
  def __init__(self, number_of_objects, probes, responses, size_of_field = 64, on_off_prob = (.99, 0.1), om = None):
    '''
    model_z(number_of_objects, probes, responses, size_of_field = 64, on_off_prob = (.99, 0.1), object_map = None)
    probes ~ P x size_of_field x size_of_field binary masks, e.g., a probe.masks array
    responses ~ T x 1 array, T number of trials, this should be empirical data
    size_of_field refers to number of pixels on each side
    on_off_prob ~ (1,0) is deterministic, meaning that if a probe overlaps with an object the subject will def. report it, and won't othewise.
    object_map ~ size_of_field x size_of_field numpy array of integers between 1 and number_of_objects
    
    runs a smart model initialization based upon stirling map idea: generates all possible membership
    functions for a set of seed points (default is a 3x3 grid), then creates corresponding objects via interpolation
    tests likelihood of each. creates a object_map attribute from map with hight likelihood. then runs a grid search
    to get the alpha/beta params for the poisson_beta_binomial noise model. creates a 'noise_model' attribute using
    max lkhd. noise params.
    
    '''
    self.number_of_objects = number_of_objects
    self.resp = responses 
    self.size_of_field = size_of_field
    self.probes = probes
    self.number_of_probes = probes.shape[0]
    self.noise_model = pbb(self.number_of_objects, on_off_prob = on_off_prob)
    self.best_om = om
    
    ##incidental parameters. feel free to fuck with them
    #self._init_on_off_probs = (0.87, 0.05)  ##starting guess noise model params
    #self._noise_grid_x_dns = 20 ##density of noise model param grid, x-axis
    #self._noise_grid_y_dns = 20 ##density of noise model param grid, y-axis
    #self._noise_grid_tiny_diff = 10**-6 ##make sure off-prob strictly < on-prob
    self._print_cycle = 100 ##iterations per print
    self._seed_point_rows = 3 ##these determine the number/position of seed points for stirling maps
    self._seed_point_cols = 3
#    self._cluster_pref = cluster_pref ##will the select the best of all possible stirling maps
#    self._map_func = mapping_function ##in case you want to use parallel mapping for intensive parts
    
    
    ##smart initialization
    #self.model_initialization()  ##creates attribute "best_om" and "noise_model"
    
    
  #def model_initialization(self):
    #'''	
    #generates multiple stirling maps and searches for one with highest likelihood.
    #then runs grid search on on_prob/off_prob
    #run as part of __init__
    #creates attributes "noise_model" and "best_om"
    #'''
    
    ###generate all stirling maps by creating a regular grid of seed points, assign all possible memberships, and interpolating
    #print 'constructing stirling maps...'
    #seed_point_rows = self._seed_point_rows
    #seed_point_cols = self._seed_point_cols
    #spm = sparse_point_maps(seed_point_rows, seed_point_cols, self.size_of_field, cluster_pref=self._cluster_pref, number_of_clusters = self.number_of_objects)
    #maps = spm.nn_interpolation()
    
    
    ###create noise grid
    #on_prob, off_prob = self.noise_grid()
	    
    ###find best stirling maps
    #cur_idx = 0
    #best_candidate_idx = None
    #while 1:
		#print 'evaluating stirling maps...'
		#candidate_lkhd = self._map_func(lambda m: self.sum_log_lkhd(m), maps)
		#cur_idx = np.argmax(candidate_lkhd)
		#print 'cur best: %d' %(cur_idx)
		#print 'cur best: %f' %(candidate_lkhd[cur_idx])
		#if cur_idx == best_candidate_idx:
			#self.best_om = np.squeeze(maps[best_candidate_idx,:,:])
			#break
		#else:
			#best_candidate_idx = cur_idx
			##print 'best: %d' %(best_candidate_idx)
			#cur_om = np.squeeze(maps[best_candidate_idx,:,:])
			###optimize alpha/beta
			#print 'evaluating noise models'
			#candidate_lkhd = [pbb(self.number_of_objects,on_off_prob = o_f).sum_log_likelihood(self.resp, self.overlap(cur_om)) for o_f in zip(on_prob,off_prob)]   
			#best_noise_params_idx = np.argmax(candidate_lkhd)
			###attach a poisson_beta_binomial instance with optimal on_prob/off_prob
			#self.noise_model = pbb(self.number_of_objects,on_off_prob = (on_prob[best_noise_params_idx], off_prob[best_noise_params_idx]))
    
    
  def predict_response(self, new_probe):
    '''
    expected value (I think?) of response given underlying object_map/probe 
    '''
    return pbb(self.number_of_objects,on_off_prob = (1, 0)).sample(self.overlap(self.best_om, probes=new_probe))
        
    

  def overlap(self, one_map,probes=None):
    '''
    overrides overlap(probes,map)
    
    overlap() applies all probes to current best object map
    overlap(probes = new_probes) applies some new probes to the current best object map
    returns a  number_of_probes x number_of_objects array
    each column is a bit string showing which of the objects a probe overlaps
    '''
    if probes is None:
      probes = self.probes
    return overlap(probes, one_map)
 
  def sum_log_lkhd(self, one_map, on_off_prob = None):
  	'''
  	sum log lkhd of a bunch of responses to a map
  	sum_log_lkhd(map)
  	returns one number
  	'''
  	return self.noise_model.sum_log_likelihood(self.resp, self.overlap(one_map), on_off_prob)
  	
