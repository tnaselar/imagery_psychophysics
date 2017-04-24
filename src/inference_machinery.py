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
from imagery_psychophysics.src.counting_machinery import window, collection_of_windows, uniquify
from time import time

def format_experimental_obsevations(experimental_observation):
    '''
    make sure all of the window names obey the sorting rules established by the "window" object.
    returns exp_obs = {(window):data, ...}
    '''
    exp_obs = dict(zip( [ window(key).tup for key in experimental_observation.keys() ], experimental_observation.values())) 
    return exp_obs

class nbd_data(object):
  '''
  nbd_data(target, experimental_observation, carryover_windows=None)
  target = an integer naming a target window
  experimental_observation ~  a dictionary like {(window):data, (window):data}, where (window) is a tuple naming the window, and data is like a response.
			      all of the window names obey the sorting rules established by the "window" object.
  carryover_windows ~ optional collection of windows to add the current nbd. important for marginalizing.
  '''
  def __init__(self, target, exp_obs, carryover_windows=None):
    
    ##the window tuples for all observations that include the target window
    nbd_obs = collection_of_windows([obs for obs in exp_obs.keys() if target in obs])  
    
    ##glue data back to the windows in the nbd_obs:
    ##NOTE: not sure how this handles case of multiple data points for a single window ??
    resp_dict = dict((no,exp_obs[no]) for no in nbd_obs.tups)
    
    #get unit windows in nbs_obs + carried-over windows from other iterations.
    try:
      nbd_windows = collection_of_windows(uniquify(nbd_obs.tups+carryover_windows.tups)).reduce2window()
    except:
      nbd_windows = collection_of_windows(uniquify(nbd_obs.tups)).reduce2window()
    
    ##all observations are included in the powerset of nbd_windows
    ##any subset of this powerset that includes the target will be summed out.
    nbd_windows_powerset = nbd_windows.powerset(nonempty=True)
    
    #these observations will not be summed out because they do not include the target window
    not_summed = collection_of_windows([wu for wu in nbd_windows_powerset.tups if target not in wu])
    
    self.target = target  
    self.resp_dict = resp_dict
    self.nbd_windows = nbd_windows
    self.not_summed = not_summed  ##a collection of windows
    
    
## a function for creating a sequence of nbd observations
def create_sequence_of_nbd_obs(target_sequence, experimental_observations):
  carryover_windows = []
  sequence = []
  for targ in target_sequence:
    
    ##construct the observations for the current nbd.
    cur_obs = nbd_data(targ, experimental_observations,carryover_windows=carryover_windows)
    
    ##add to the sequence
    sequence.append(cur_obs)
    
    ##get the windows that will carryover in this sequence
    carryover_windows = cur_obs.not_summed
    
    ##remove the observations that will have been summed out at this stage
    [experimental_observations.pop(co) for co in cur_obs.resp_dict.keys()]
    
  return sequence

##a function for counting all of the colorings consistent with a sequence of nbd observations
def coloring_messenger(n_colors,nbd_sequence,sample_sequence,counter_sequence,doprint=True):
    '''
    coloring_messenger(nbd_sequence,sample_sequence,counter_sequence)
    we expect nbd_sequence, sample_sequence, and counter_sequence to be registered!
    
    Let nbd,samples,counter be registered elements from each sequence.
    Then:
    samples.columns = nbd.nbd_windows.powerset(nonempty=True).strings
                    = counter.nbd.strings
    '''
    collect_messages = []
    message = 1
    carryover_windows = None
    ##loop to count
    for nbd,samples,counter in zip(nbd_sequence,sample_sequence,counter_sequence):
        start = time()
        print '===============target window: %d' %(nbd.target)

        if doprint:
            print 'current observations: %s' %(nbd.resp_dict,)
        
        ##------merge current samples with previous message
        try:
            message_board = samples.merge(message,how='right', on=list(carryover_windows.strings))         
        except:
            ##assuming  constant message
            message_board = samples
            message_board['message'] = message
            ##this should mean it's the first iteration
            if doprint:
                print 'window: %s, assuming constant or no message' %(nbd.target)

        ##------apply counting function
        counting_func = lambda row: counter.count_consistent_coloring(n_colors,row[nbd.nbd_windows.powerset(nonempty=True).strings])
        
        count_start = time()
        
        message_board['count'] = message_board.apply(counting_func,axis=1)
        if doprint:
            print 'finished counting in %f seconds' %(time()-count_start)
            
        ##------take the the product of the computed functions (ie., count, message)
        try:
            message_board['product'] = message_board[['count','message']].prod(axis=1)
        except KeyError:
            print 'no message to multiply'
            message_board['product'] = message_board[['count']].prod(axis=1)


        ##------sum to construct message, 
        try:
            message = message_board.groupby(nbd.not_summed.strings,as_index=False)['product'].sum() 
            message.rename(columns={'product':'message'},inplace=True)
            collect_messages += [message_board]
            carryover_windows = nbd.not_summed
        except ValueError:
            print 'window %s, no groupby. summing' %(nbd.target)
            message = message_board['product'].sum()
            collect_messages += [message_board]
        
        print 'elapsed time, iteration %d: %f' %(nbd.target, time()-start)

    print '*******final message is: %f' %(message)
    return message, collect_messages   

##a function for counting the colorings consistent with many difference color count sequences, independently
def indie_coloring_messenger(n_colors,nbd_sequence,sample_sequence,counter_sequence,doprint=True):
    '''
    indie_coloring_messenger(nbd_sequence,sample_sequence,counter_sequence)
    we expect nbd_sequence, sample_sequence, and counter_sequence to be registered!
    
    Let nbd,samples,counter be registered elements from each sequence.
    Then:
    samples.columns = nbd.nbd_windows.powerset(nonempty=True).strings
                    = counter.nbd.strings
    '''
    collect_messages = []
    message = 1
    
    ##loop to count
    for nbd,samples,counter in zip(nbd_sequence,sample_sequence,counter_sequence):
        start = time()
        print '===============target window: %d' %(nbd.target)

        if doprint:
            print 'current observations: %s' %(nbd.resp_dict,)
        
        ##------merge current samples with previous message
	message_board = samples.copy()
	message_board['message'] = message
	##this should mean it's the first iteration

        ##------apply counting function
        counting_func = lambda row: counter.count_consistent_coloring(n_colors,row[nbd.nbd_windows.powerset(nonempty=True).strings])
        
        count_start = time()
        
        message_board['count'] = message_board.apply(counting_func,axis=1)
        if doprint:
            print 'finished counting in %f seconds' %(time()-count_start)
            
        ##------take the the product of the computed functions (ie., count, message)
        try:
            message_board['message'] = message_board[['count','message']].prod(axis=1)
            message = message_board[['message']]
            collect_messages += [message_board]
        except KeyError:
            print 'no message to multiply'
            message_board['product'] = message_board[['count']].prod(axis=1)


        ##------sum to construct message, 
        #try:
            #message = message_board[['product']].rename(columns={'product':'message'},inplace=True)
            #collect_messages += [message]
        #except ValueError:
	    #print 'not sure what is happening'
            ##print 'window %s, no groupby. summing' %(nbd.target)
            ##message = message_board['product'].sum()
            ##collect_messages += [message]
	  
        #print 'elapsed time, iteration %d: %f' %(nbd.target, time()-start)
    if doprint:
      print '*******final message is: %s' %(message.values,)
    return message, collect_messages   
