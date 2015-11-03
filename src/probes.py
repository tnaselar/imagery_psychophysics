##simple class for generating circular probes. these are like dead leaves, they don't combine to form proper object maps

class probes(object):
  def __init__(self, probe_rad=6, probe_step=3, size_of_field=64):
    '''
    probes(probe_rad=6, probe_step=3, size_of_field=64)
    construct a regular, square grid of probes for the experiment
    probe_rad ~ radius of probe in pixels
    probe_step ~ spacing between probes in pixels
    size_of_field ~ number of pixels on each side of the square visual field
    
    attributes: 
    npix = size_of_field**2
    num_probes = number of probes, always with an integer square root
    mask ~ sqrt(num_probe) x sqrt(num_probes) x size_of_field x size_of_field: binary masks for each probe
    
    
    '''
    self.size_of_field = size_of_field
    self.npix = size_of_field**2
    self.construct_probes(probe_rad, probe_step)
    
  def construct_probes(self, probe_rad, probe_step):
    from numpy import zeros, array, dstack, meshgrid, arange
    image_size = self.size_of_field
    pixel_index_i, pixel_index_j = meshgrid(range(image_size), range(image_size))
    num_probe_rows = len(arange(0,image_size, probe_step))
    self.num_probes = num_probe_rows**2
    PROBES = zeros((num_probe_rows,num_probe_rows, image_size, image_size))
    for probe_index_i, probe_location_i in enumerate(arange(0,image_size,probe_step)):
      for probe_index_j, probe_location_j in enumerate(arange(0,image_size,probe_step)):
	  pix_i, pix_j = self.__pairwise_less_than__(array([probe_location_i, probe_location_j]), dstack([pixel_index_i, pixel_index_j]), probe_rad)
	  PROBES[probe_index_i,probe_index_j, pix_i, pix_j] = 1
    self.masks = PROBES

  def __pairwise_less_than__(self, ref_point, all_other_points, radius):
    from numpy import sqrt, nonzero, sum
    return nonzero(sqrt(sum((ref_point - all_other_points)**2,axis=2)) <= radius)

  

    
  ##create set of overlapping circular probes of fixed radius
#probe_rad = 6 ##radius in pixels
#probe_step = 3 ##spacing between probes
#nbd_rad = 5 ##nbd radius in units of probe size 
#nbd_step = 1##nbd step in units of probe size

#image_size = int(np.sqrt(dl.npix))

###generate explicit indices for all pixels in an object map
#pixel_index_i, pixel_index_j = meshgrid(range(image_size), range(image_size))
#num_probe_rows = len(arange(0,image_size, probe_step))
#num_probes = num_probe_rows**2
#PROBES = zeros((image_size, image_size, num_probe_rows,num_probe_rows))
#for probe_index_i, probe_location_i in enumerate(arange(0,image_size,probe_step)):
    #for probe_index_j, probe_location_j in enumerate(arange(0,image_size,probe_step)):
        #pix_i, pix_j = pairwise_less_than(array([probe_location_i, probe_location_j]), dstack([pixel_index_i, pixel_index_j]), probe_rad)
        #PROBES[pix_i,pix_j,probe_index_i,probe_index_j] = 1