import numpy as np
import pandas as pd
import warnings
import re
from itertools import combinations, chain, product
from math import factorial as bang
from math import pow
from scipy.misc import comb
from imagery_psychophysics.src.stirling_maps import stirling_partitions as stp
from imagery_psychophysics.src.stirling_maps import stirling_num_of_2nd_kind as snk
from imagery_psychophysics.src.stirling_maps import sparse_point_maps as spm
from bitarray import bitarray
from sklearn.utils.extmath import cartesian
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter
from imagery_psychophysics.src.model_z import noise_grid
from numpy.random import binomial
from object_parsing.src.image_objects import apply_mask
from time import time


class window(object):
    '''
    tup ~ (1,2,3,...) a collection of integers
    string ~ '1,2,3,4...' a comma-separated string
    window(collection_of_integers)
    window.string
    window.tup
    window.powerset(nonempty=False, string=False) ~ returns a set containing all possible subsets of integers
    '''
    def __init__(self,collection_of_integers):
        if type(collection_of_integers) is str:
            self.string = self.sort(collection_of_integers)
            self.tup = self.str2tup(self.string)
        else:
            self.tup = self.sort(collection_of_integers)
            self.string = self.tup2str(self.tup)
    
    
    def powerset(self,nonempty=False, strict=False):
        return collection_of_windows(self.__powerset__(nonempty, strict))
    
    def __powerset__(self,nonempty=False, strict=False):
        '''
        powerset(nonempty=False,string=False)
        return a set of tuples (or comma-separated strings) encoding all possible subsets of the integers
        if nonempty=True, don't return the empty set
        if string = True, return a set of comma-separated strings instead of tuples
        if strict = True, don't return the original tuple (string) as a subset
        NOTE: regardless of how the collection of input integers are ordered or formatted, they will be sorted.
        '''
        
        if strict:
            pwr = self.__powerset__(nonempty=nonempty, strict=False)
            pwr.discard(self.tup)
            return pwr
        if nonempty:
            pwr = self.__powerset__(nonempty=False)
            pwr.discard(())
            return pwr
        else:
            s = list(self.tup)
            return set(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    
    def str2tup(self,string_of_integers):
        return tuple(map(int,string_of_integers.split(',')))
        
    def tup2str(self,tuple_of_integers):
        return ','.join(map(str,tuple_of_integers))
    
    def sort(self,collection_of_integers):
        if type(collection_of_integers) is str:
            return self.sort(self.str2tup(collection_of_integers))
        else:
            return tuple(sorted(collection_of_integers))
	  
	  
class collection_of_windows(object):
    '''
    a set of tuples {(), ..., ()}. 
    alternatively, may also be a set of comma-separated strings: {'1,2,3', '1', '4,5,6'}
    integers in each tuple/string are sorted as in window.object. no redundancy.
    '''

    def __init__(self, collection):
        self.tups = list(uniquify([window(w).tup for w in collection]))
        self.strings = list(uniquify([window(w).string for w in collection]))
        
    ##reducing the collection of windows to a single window by extracting the unique integers
    def reduce2window(self):
        '''
        reduce2window() 
        returns a window object created from the unique of integers in the collection
        '''
        return  window(tuple(np.unique([kk for tup in self.tups for kk in tup])))
          

##a uniquifying function
def uniquify(seq):
  # order preserving
  noDupes = []
  [noDupes.append(i) for i in seq if not noDupes.count(i)]
  return noDupes
 

##all possible t x n binary matrices with non-empty rows
##yields a generator. doesn't take any time to evaluate.
def generate_window_colorings(n,t, bits = False):

  # 2^(n-1)  2^n - 1 inclusive
  bin_arr = range(0, int(pow(2,n)))
  bin_arr = [bin(i)[2:] for i in bin_arr]

  # Prepending 0's to binary strings
  max_len = len(max(bin_arr, key=len))
  bin_arr = [i.zfill(max_len) for i in bin_arr]
  bin_arr.remove('0'*n)
  if bits:
      bin_arr = map(bitarray, bin_arr)
  return product(bin_arr,repeat=t)

##count number of colors in the union of one or more atoms. atom colorings are represented as bitarrays
def multi_union(iterable):
    '''multi_union(list_of_bitarrays)'''
    bs = bitarray('0'*len(iterable[0]))
    for nxt in iterable:
        bs |= nxt
    return bs.count()
    

##    
def nbd_color_counts(window_nbd, window_colorings,expand=True):
    '''
    given a set of window colorings (encoding as n-bit strings) for a nbd with t windows, return the color count for all subsets in the ndb.
    (window_nbd, color_counts) = nbd_color_counts(window_nbd, window_colorings)
    '''
    
    if expand: 
    ##assume only the nbd has been provided, expand to obtain the collection of all windows made from all subsets of the nbd
      nbd_powerset = window_nbd.powerset(nonempty=True)
      nbd = window_nbd
      
      
    ##otherwise assume we already have the powerset expansion, and just rename
    else:
      nbd_powerset = window_nbd
      nbd = nbd_powerset.reduce2window()
      
    
    ##each coloring gets a window name
    window_colorings = dict(zip(nbd.tup, window_colorings))
    
    ##package the colorings together for union evaluation
    pwr = map(lambda idx: [window_colorings[ii] for ii in idx], nbd_powerset.tups)
    unions = map(multi_union, pwr)
    return nbd_powerset,unions    


def enumerate_nbd_color_counts(n_colors, window_nbd, redundant=False):
    
    '''
    enumerate_nbd_color_counts(n_colors, window_nbd, redundant=False)
    window_nbd is a window object
    enumerates all possible color counts for the given window nbd
    returns a pandas dataframe, columns = window_nbd.powerset(nonempty=True).strings
    Uses matrix multiplication to get unions.
    '''
    n_win = len(window_nbd.tup)
    theory = (2**n_colors-1)**n_win
    
    chunksize = 1000000
    
    
    window_names = window_nbd.powerset(nonempty=True)
    unions = []
    cntr = 0
    start = time()
    
    ##===WORKING ON THIS !!!=======
    coeffs = np.zeros((n_win,len(window_names.tups)))
    coeffs = pd.DataFrame(data = coeffs, columns = window_names.tups, index = window_nbd.tup)
    for t in window_names.tups:
        coeffs.loc[t,t] = 1
    ###===========================
    
    for win_clrs in generate_window_colorings(n_colors,n_win,bits=False):
        if not np.mod(cntr,chunksize):
	  
	    ##inform.
            print 'counts so far: %d' %(cntr)
            print 'took %f seconds' %(time()-start)
           
            
            ##accum. current
	    coloring_array = np.array(map(lambda x: np.fromstring(x,dtype='u1')-ord('0'),win_clrs))
	    unions += [np.sum(coloring_array.T.dot(coeffs).clip(0,1),axis=0)]
	    
	    ##enlarge df
            try:
	      ##if not the first round
	      df = df.append(pd.DataFrame(np.array(unions), columns = window_names.strings))
	    except:
	      try:
		##if the first round
		df = pd.DataFrame(np.array(unions), columns = window_names.strings)
	      except:
		print 'this does not work'
	    
	    ##deal with redundancy
	    if not redundant:
	      #print 'deduplicating'
	      df = df.drop_duplicates()
	    
	    ##reset unions,timers
	    unions = []
	    start = time()
        ##otherwise just keep accumulating
        else:    
	  coloring_array = np.array(map(lambda x: np.fromstring(x,dtype='u1')-ord('0'),win_clrs))
	  unions += [np.sum(coloring_array.T.dot(coeffs).clip(0,1),axis=0)]
        cntr += 1
    
    ##append last batch
    if len(unions):
      df = df.append(pd.DataFrame(np.array(unions), columns = window_names.strings))
    if not redundant:
	#print 'deduplicating'
	df = df.drop_duplicates()

    ##inform
    print 'took %f seconds' %(time()-start)
    print '---'
    print 'should be: %d. Is: %d' %(theory, cntr)
    print 'reduced to: %d' %(df.shape[0])
    print 'reduction is: %f percent' %(100-df.shape[0]/float(theory)*100)


    
    ##finis
    return df
    
  
     
class consistent_map_counter(object):
  '''
  upon init, creates the "counting coefficients".
  returns a number of maps consistent with that color count
  consistent_map_counter(num_objects,target, window_names)
  num_objects = # objects in the universe
  target = the name of the target (unit) window. should be a tuple like (1,)
  window_names = collection_of_windows object. used to interpret what each color count belongs to.
 
  '''
  

  def __init__(self, target, window_names):
    nbd_minus_target = list(window_names.reduce2window().tup)
    nbd_minus_target.remove(target)
    nbd_minus_target = window(nbd_minus_target).powerset(nonempty=True)  ##<<this is expensive.
    self.target = target
    self.nbd = window_names
    self.nbd_minus_target = nbd_minus_target
    self.upper_counting_coeffs = self.generate_counting_coeffs(nbd_minus_target)  ##T x S, T = S
    self.lower_counting_coeffs = self.generate_counting_coeffs(window_names)	  ##T x S, T > S
  
  ##nbd is a collection_of_windows object
  def generate_counting_coeffs(self,nbd):
    aTS = np.zeros((len(nbd.tups),len(self.nbd_minus_target.tups)+1)) ##the +1 is for the empty set
    S_nbd = self.nbd_minus_target.reduce2window().tup  ##<<this will give all the unit windows in this nbd.
    T_nbd = nbd.reduce2window().tup
    for s,S in enumerate(self.nbd_minus_target.tups+[()]): ##notice we add the empty set explicitly as the last col.
      if S == S_nbd:  ##<<enforce strict subset
	continue
      for t,T in enumerate(nbd.tups):
	if len(set(S).difference(T)) <= 0:
	  aTS[t,s] = (-1)**(len(S)+len(T)+1)
    return aTS
  
  ###nbd_color_count MUST be ordered according to "window_names"
  def count_consistent_coloring(self,num_objects,nbd_color_count):
    nbd_dict = dict(zip(self.nbd.tups,nbd_color_count))  ##<<user must know how to order the "nbd_color_counts" !!!

    upper_color_counts = np.array([nbd_dict[w] for w in self.nbd_minus_target.tups])
    upper_counts = upper_color_counts.dot(self.upper_counting_coeffs) ##1 x S 

    lower_counts = np.array(nbd_color_count).dot(self.lower_counting_coeffs)    ##1 x S
    count  = np.prod(map(lambda up,down: comb(up,down,exact=True), upper_counts,lower_counts))
    
    ##deal with last case
    max_union = self.nbd.reduce2window().tup
    max_union_size = nbd_dict[max_union]
    max_union_no_targ = self.nbd_minus_target.reduce2window().tup
    if max_union_no_targ:
      max_union_size_no_targ = nbd_dict[max_union_no_targ]
    else:
      max_union_size_no_targ = 0
    count *= comb(num_objects-max_union_size_no_targ, max_union_size-max_union_size_no_targ,exact=True)
    return count
  
  
  
  ###the spatial parts
  def count_consistent_maps(self,num_objects,nbd_color_count,target_window_size):
    nbd_dict = dict(zip(self.nbd.tups,nbd_color_count)) ##<<user must know how to order the "nbd_color_counts" !!!
    target_color_count = nbd_dict[(self.target,)]
    return self.count_consistent_coloring(num_objects,nbd_color_count)*snk(target_window_size,target_color_count)*bang(target_color_count)
    
