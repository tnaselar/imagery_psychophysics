import PIL
import numpy as np

def nearest_point_to(a, ref):
	d = ((a-ref)**2).sum(axis=1)  # compute distances
	ndx = d.argsort() # indirect sort 
	return a[ndx[0]]

def angular_distance(many_angles, ref_angle):
	many_angles = np.atleast_1d(many_angles)
	out_of_bounds = ((many_angles >= 360) | (many_angles < 0))
	many_angles[out_of_bounds] = np.mod(many_angles[out_of_bounds],360)
	if ref_angle >= 360 or ref_angle < 0:
		ref_angle = np.mod(ref_angle,360)
	ab = np.abs(many_angles-ref_angle)
	ab[ab>=180] = 180-np.mod(ab[ab>=180],180)
	return ab

def cart2polar(pts):
	from numpy import angle,abs,concatenate
	shape = (pts.shape[0],1)
	return abs(pts[:,0]+1j*pts[:,1]).reshape(shape), angle(pts[:,0]+1j*pts[:,1]).reshape(shape)
	

def com(pts):
	from numpy import sum
	return sum(pts,axis=0).astype('float')/pts.shape[0]
	

	

##So it goes
#(1) create dead leaves by calling "map_as_cols" in the __init__method of "dead_leaves"
#(2) spawn an object_map (this should also use "map_as_cols" object)
#(3) break off the map_as_cols objects and pass to object map prior
#(4) the object map prior will use "pull" and remove as it recurses

class dead_leaves_prior(object):
  
  def __init__(self):
    print 'initiating dead leaves prior'
  
  def inclusion_probability(self,one_object, leaves):
    n_leaves = leaves.count_myself()
    return np.sum(leaves.find_match(one_object)).astype('float')/n_leaves
  
  def evaluate_probability(self, om, leaves):
    K = om.count_myself()
    if K == 0:
        #print "finished"
        return 1
    else:
        try:
            normalization_constant = 1./(1 - self.inclusion_probability(om.pull(0), leaves))
        except:
            1/0      
        prob = 0
       # print om.count_myself()
        for ii in range(1,K+1):
           prob += self.inclusion_probability(om.pull(ii),leaves)*self.evaluate_probability(om.remove(om.pull(ii)),leaves.remove(om.pull(ii)))
            
    return normalization_constant*prob


##the basic 
class binarized_object_map(object):
  def __init__(self, data):
    from numpy import array
    try: ##maybe it's an image
      self.bom = self.__convert_from_image__(data)
      self.npix = self.bom.shape[0]
    except: ##otherwise assume its an array in the proper format
      self.bom = data
      self.npix = self.bom.shape[0]
    
  
  def __convert_from_image__(self, map_as_image):
    from numpy import unique, array, ravel, hstack
    from object_parsing.src.image_objects import mask_quantize
    object_idx = unique(array(map_as_image))
    npix = array(map_as_image).size
    tmp_array = array([]).reshape((npix,0))
    for odx in object_idx:
      tmp_array = hstack([tmp_array,ravel(array(mask_quantize(map_as_image,odx,scale=1))).reshape((npix,1))])
    return tmp_array
  
  ##count the number of objects in the map
  def count_myself(self):
    return self.bom.shape[1]
  
  ##add another object to the map?
  def append(self, data):
    from numpy import concatenate
    try:
      self.bom = concatenate([self.bom, self.__convert_from_image__(data)],axis=1)
    except:
      self.bom = concatenate([self.bom, data],axis=1)
    self.npix = self.bom.shape[0]
  
  ##grab a single object from the object map
  def pull(self, n):
    if n:
        foo = np.atleast_2d(self.bom[:,n-1]).T  ##<<NOTE we are dealing with a pointer here
    else:
	foo =  np.zeros((self.npix,1))
    return binarized_object_map(foo)

  ##remove all an object from the object map, and kill all pixels associate with it.
  ##NOTE: after removing an object, the bom cannot be converted back into a proper image
  def remove(self, pulled_bom):
	idx = np.array(pulled_bom.bom).astype('bool').flatten()  ## << make sure
	foo = self.bom[~idx, :]
	indices = foo.any(axis=0)
	return binarized_object_map(foo[:,indices])

  ##is there an object in the bom that matches the input object?   
  def find_match(self, one_object):
    return np.array(one_object.bom==self.bom).all(axis=0)
    
  def subsample(self, pix_dx):
     ##removes all but the named pixels from bom. returns new bom instance with subsampled bom bound to original
     ##this may convert some existing object to "null" objects, so those are removed
    bool_indx = np.ones(self.npix)
    bool_indx[pix_dx] = 0 ##<<this means we will remove everything *but* the selected pixels
    return self.remove(binarized_object_map(bool_indx)) ##<<not properly formatted as a col, but shouldn't matter   

    
##a collection of boms. creates a big binary matrix from the columns of a bunch of boms
##keeps track of which columns belong to what map. 
##allows you to spawn a proper object_map object given an index.
##TODO: Initialize using a matrix of object maps in addition to the file_list method
class dead_leaves(binarized_object_map):  
    def __init__(self, object_map_file_list, image_shape=(32,32)):
	self.file_list = object_map_file_list
	self.image_shape = image_shape
	tmp_image = PIL.Image.open(object_map_file_list[0])
	tmp_image.thumbnail(image_shape)
	super(dead_leaves, self).__init__(tmp_image)
	tmp_num = self.count_myself()
	self.map_list = [[0,tmp_num]]
	for ff in range(1, len(object_map_file_list)):
	  if not ff%100:
	    print 'remaining: %d' %(ff)
	  tmp_image = PIL.Image.open(object_map_file_list[ff])
	  tmp_image.thumbnail(image_shape)
	  self.append(tmp_image)
	  self.map_list.append([tmp_num,self.count_myself()])
	  tmp_num = self.count_myself()

	
    def spawn_object_map(self,om_dx):
      '''
      spawn_object_map(om_dx):
      supply an index for the object map you want to see, return an object_map instance
      '''
      as_cols = self.bom[:,slice(self.map_list[om_dx][0], self.map_list[om_dx][1])]
      as_cols = as_cols*range(1,as_cols.shape[1]+1)
      as_cols = np.sum(as_cols,axis=1)
      as_cols = PIL.Image.fromarray(as_cols.reshape(self.image_shape).astype('int8'), mode='L')
      return object_map(as_cols)
    
    def overlap(self, binary_masks):
      '''
      overlap(binary_masks)
      pass a collection of binary masks such as the "masks" attribute of a "probes" instance
      will return a report on which of the probes overlaps (even partially) with which of the objects
      output is a list of arrays~num_objects x number_of_masks
      '''
      from numpy import array
      number_of_masks = binaray_masks.shape[2]
      RESPONSES = dl.bom.T.dot(self.masks.reshape(dl.npix,number_of_masks))  ##shape = num_objects x num_probes
      return map(lambda x: RESPONSES[slice(x[0],x[1]), :]>0, dl.map_list)


        
## a subclass with a bunch of methods for manipulating and analyzing a bom 
class object_map(binarized_object_map):
	
	def __init__(self, object_map_image):
		self.map_as_image = object_map_image
		self.map_as_array = np.array(self.map_as_image)
		self.npix = self.map_as_array.size
		self.shape = self.map_as_array.shape
		binarized_object_map.__init__(self, object_map_image)
		self.map_as_membership = self.__mem__()
		self.HARD_BIN_WIDTH = 4 ##HACK: this seems like a fairly unimportant parameter. see "boundary_points"	
	
	def pull(self,odx, format='cols'):
		from object_parsing.src.image_objects import mask_quantize
		from numpy import ravel, array		
		if format=='cols':
			return super(object_map, self).pull(odx) ##returns a bom instance
		elif format == 'bool':
			return super(object_map, self).pull(odx).bom.astype('bool').flatten() ##boolean index array
		elif format == 'image':
			return mask_quantize(self.map_as_image,odx,scale=1)
	
	def mask_to_points(self, odx):
		from numpy import meshgrid, concatenate, sum
		ix, iy = meshgrid(range(self.shape[0]),range(self.shape[1]))
		point_mask = self.pull(odx,format='bool')
		shape = (sum(point_mask),1)
		return concatenate([ix.ravel()[point_mask].reshape(shape),iy.ravel()[point_mask].reshape(shape)], axis=1)
			
	def __mem__(self):
		return [self.mask_to_points(odx) for odx in range(1,self.count_myself()+1)]
	
	def find_min_to(self, odx, point):
	  '''
	  find_min_to(odx, point):
	  returns the point in the object closest to specified point
	  '''
	  return nearest_point_to(self.map_as_membership[odx-1], point)

	def number_of_pixels(self, odx):
	  '''
	  number_of_pixels(odx)
	  number of pixels occupied by object odx
	  '''
	  return np.sum(self.pull(odx).bom)
	
	def percent_of_map(self, odx):
	  '''
	  percentage of map occupied by object odx
	  percent_of_map(odx)
	  '''
	  return np.array(self.number_of_pixels(odx)).astype('float')/self.npix
		
	def object_com(self,odx):
	  '''
	  object_com(odx)
	  spatial center of mass of object. may be part of other object
	  '''
	  return com(self.map_as_membership[odx-1])
				
	
	def object_center(self, odx):
	##find point closest to center of mass of object
	  return self.find_min_to(odx, self.object_com(odx))
		
	def boundary_points(self, odx, n_points):
	  '''##TODO: Doesn't check for repeats, which happen OFTEN. # of unique pts ~= n_points!!!'''
	  from math import pi
	  if not n_points:
		  pass
	  else:
		  
		  theta_delta = 360./n_points
		  oc = self.object_center(odx)
		  r,theta = cart2polar(self.map_as_membership[odx-1]-oc)
		  theta = theta*(180/pi)
		  max_dx = r.argmax()
		  boundary_r = r[max_dx]
		  boundary_theta = theta[max_dx]
		  
		  for ii in range(1,n_points):
			  new_dx = np.concatenate(angular_distance(theta, theta[max_dx]+ii*theta_delta)).argsort()
			  hbw = min(self.HARD_BIN_WIDTH, new_dx.size)
			  new_new_dx = r[new_dx[0:hbw]].argmax()
			  boundary_r = np.concatenate([boundary_r, r[new_dx[new_new_dx]]])
			  boundary_theta = np.concatenate([boundary_theta, theta[new_dx[new_new_dx]]])
	  
	  return np.concatenate([np.atleast_2d(boundary_r*np.cos(pi/180.*boundary_theta)+oc[0]).T, np.atleast_2d(boundary_r*np.sin(pi/180.*boundary_theta)+oc[1]).T],axis=1)
			
	def good_samples(self, n_samples):
		from numpy import array, concatenate, atleast_2d
		##distribute the sample among objects. this only makes sense here
		def distribute(numer, denom):
		  from numpy import remainder, floor
		  rem = remainder(numer, denom)
		  base = floor(numer/denom)
		  return [base+(ii<rem) for ii in range(denom)] 
		if n_samples < self.count_myself():
		  print 'need at least one sample per object. number samples is set to %d', self.count_myself()
		n_samples = max(n_samples, self.count_myself())
		number_samples_per_object = distribute(n_samples, self.count_myself())
		number_samples_per_object.sort() ##<<put in ascending order
		order_by_size = np.array([self.number_of_pixels(ii) for ii in range(1,self.count_myself()+1)]).argsort()+1
		#order_by_size = range(1,len(number_samples_per_object)+1)
		samples = array([]).reshape((0,2)) ##<<just to get us started
		cnt = 1
		
		for ii,jj in zip(order_by_size, number_samples_per_object):
		  samples = np.concatenate([samples,np.atleast_2d(self.object_center(ii))],axis=0)
		  if jj-1 > 0:
		      samples = np.concatenate([samples,self.boundary_points(ii,int(jj)-1)],axis=0)
		return samples ##TODO: remove the (0,0) at the top of the array!!

			
	def points_2_indices(self,pts):
	  '''accepts points in image ij coors'''
	  foo = []
	  for ii in range(pts.shape[0]):
	    foo.append(np.ravel_multi_index(tuple(pts[ii,[1,0]].astype('int64')), self.map_as_image.size)) ##<<Note the conversion from i,j to x,y coors.
	  return np.array(foo)
			
		
	def indices_2_pts(self,idx):
	  '''returns points in image ij coors.'''
	  return np.array([np.unravel_index(ii, self.map_as_image.size) for ii in idx])
	  
		
		
	def overlap(self, binary_masks):
	  '''
	  overlap(binary_masks)
	  pass a collection of binary masks such as the "masks" attribute of a "probes" instance
	  binaray_masks is number_of_masks_rows x number_of_masks_cols x npix x npix 
	  will return a report on which of the probes overlaps (even partially) with which of the objects
	  output is num_objects x number_of_masks binary array with 1's for each overlap
	  '''
	  from numpy import array
	  number_of_masks = binary_masks.shape[0]#*binary_masks.shape[3]
	  RESPONSES = binary_masks.reshape(number_of_masks, self.npix).dot(self.bom)  ##shape = num_objects x num_probes
	  return RESPONSES>0
	
		
		
		
		
		
		
		
		
		
		
	
