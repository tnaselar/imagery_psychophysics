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
	

##TODO: read in a bunch of images, immediately thumbnail them, convert to an object_map, format_as_cols, concatenate to matrix
##TODO: methods: "remove":removes pixels and all null objects.
##TODO: methods: "unique": ratio of unique to total number of objects
#class dead_leaves(object):
  
  def __init__(self, object_map_file_list, size=(32,32)):
    self.file_list = object_map_file_list
    self.object_array = object_map(PIL.Image.open(object_map_file_list.pop()).thumbnail(size)).map_as_cols
    while object_map_file_list:
      self.object_array = np.concatenate([np.object_array,object_map(PIL.Image.open(object_map_file_list.pop()).thumbnail(size)).map_as_cols],axis=1)
      


##TODO: add a "remove" function that pulls out pixels / objects from "map_as_cols" representation
class object_map(object):
	
	def __init__(self, object_map):
		self.map_as_image = object_map
		#self.map_as_image = PIL.Image.open(object_map_file)
		self.map_as_array = np.array(self.map_as_image)
		self.npix = self.map_as_array.size
		self.shape = self.map_as_array.shape
		self.map_as_cols = self.format_map_as_cols()		
		self.number_of_objects = self.map_as_cols.shape[1]
		self.map_as_membership = self.mem()
		
	
	def pull(self,odx, format='cols'):
		from object_parsing.src.image_objects import mask_quantize
		from numpy import ravel, array		
		if format=='cols':
			return np.ravel(np.array(mask_quantize(self.map_as_image,odx,scale=1))).reshape((self.npix,1))
		elif format == 'bool':
			return np.ravel(np.array(mask_quantize(self.map_as_image,odx,scale=1))).astype('bool')
		elif format == 'image':
			return mask_quantize(self.map_as_image,odx,scale=1)
	
	def mask_to_points(self, odx):
		from numpy import meshgrid, concatenate, sum
		ix, iy = meshgrid(range(self.shape[0]),range(self.shape[1]))
		point_mask = self.pull(odx,format='bool')
		shape = (sum(point_mask),1)
		return concatenate([ix.ravel()[point_mask].reshape(shape),iy.ravel()[point_mask].reshape(shape)], axis=1)
			
	def mem(self):
		return [self.mask_to_points(odx) for odx in range(1,self.number_of_objects+1)]
	
	
	def format_map_as_cols(self):
		from numpy import unique, concatenate		
		obj_indices = unique(self.map_as_array)		
		print obj_indices
		om = self.pull(1, format='cols')
		for ii in obj_indices[1:]:
		  om = concatenate([om, self.pull(ii, format='cols')],axis=1)
		return om
	
	def find_min_to(self, odx, point):
	##returns the point in the object closest to specified point
		return nearest_point_to(self.map_as_membership[odx-1], point)

	def number_of_pixels(self, odx):
		from numpy import sum
		return sum(self.pull(odx))
	
	def percent_of_map(self, odx):
		return self.number_of_pixels(odx).astype('float')/self.npix
		
	def object_com(self,odx):
		##center of mass of object. may be part of other object
		return com(self.map_as_membership[odx-1])
				
	
	def object_center(self, odx):
	##find point closest to center of mass of object
		return self.find_min_to(odx, self.object_com(odx))
		
	def boundary_points(self, odx, n_points):
		from math import pi
		if not n_points:
			pass
		else:
			HARD_BIN_WIDTH = 100 ##HACK: <<But this seems like a really unimportant parameter.
			theta_delta = 360./n_points
			oc = self.object_center(odx)
			r,theta = cart2polar(self.map_as_membership[odx-1]-oc)
			theta = theta*(180/pi)
			max_dx = r.argmax()
			boundary_r = r[max_dx]
			boundary_theta = theta[max_dx]
			
			for ii in range(1,n_points):
				new_dx = np.concatenate(angular_distance(theta, theta[max_dx]+ii*theta_delta)).argsort()
				hbw = min(HARD_BIN_WIDTH, new_dx.size)
				new_new_dx = r[new_dx[0:hbw]].argmax()
				boundary_r = np.concatenate([boundary_r, r[new_dx[new_new_dx]]])
				boundary_theta = np.concatenate([boundary_theta, theta[new_dx[new_new_dx]]])
		
		return np.concatenate([np.atleast_2d(boundary_r*np.cos(pi/180.*boundary_theta)+oc[0]).T, np.atleast_2d(boundary_r*np.sin(pi/180.*boundary_theta)+oc[1]).T],axis=1)
			
	def good_samples(self, n_samples):
		
		##distribute the sample among objects. this only makes sense here
		def distribute(numer, denom):
		  from numpy import remainder, floor
		  rem = remainder(numer, denom)
		  base = floor(numer/denom)
		  return [base+(ii<rem) for ii in range(denom)] 

		number_samples_per_object = distribute(n_samples, self.number_of_objects)
		number_samples_per_object.sort() ##<<put in ascending order
		order_by_size = np.array([self.number_of_pixels(ii) for ii in range(1,self.number_of_objects+1)]).argsort()+1
		samples = np.zeros((1,2)) ##<<just to get us started
		for ii in order_by_size:
		  for jj in number_samples_per_object:
		    samples = np.concatenate([samples,np.atleast_2d(self.object_center(ii))],axis=0)
		    samples = np.concatenate([samples,self.boundary_points(ii,int(jj)-1)],axis=0)		    
		return samples ##TODO: remove the (0,0) at the top of the array!!

			
		
			
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	
