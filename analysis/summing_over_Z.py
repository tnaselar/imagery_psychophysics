# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd
from itertools import combinations, chain, permutations, product
from functools import partial
from itertools import combinations_with_replacement as cwr
from math import factorial as bang
from math import pow
from scipy.misc import comb
from imagery_psychophysics.src.stirling_maps import stirling_partitions as stp
from imagery_psychophysics.src.stirling_maps import stirling_num_of_2nd_kind as snk
from imagery_psychophysics.src.stirling_maps import sparse_point_maps as spm
from time import time
from bitarray import bitarray
from sklearn.utils.extmath import cartesian
from matplotlib import pyplot as plt
from PIL import Image
import warnings
import pickle
from imagery_psychophysics.src.model_z import noise_grid
from numpy.random import binomial
from object_parsing.src.image_objects import apply_mask

%matplotlib inline

# <markdowncell>

# ### Some fake data to drive testing

# <codecell>

##corner observations on a square
# observation_windows = [(1,), (2,), (3,), (4,), (1,2), (1,3), (2,4), (3,4)]

##complete observation on a square
observation_windows = [(1,), (2,), (3,), (4,), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]

##independent observations
# observation_windows = [(1,), (2,), (3,)]

##markov observations
#observation_windows = [(1,), (2,), (3,), (1,2), (2,3)]


size_of_universe = 4
nobs = len(observation_windows)
responses = np.random.randint(1,high=size_of_universe+1,size=nobs)                          ##<<may not be consistent cardinalities!!
experimental_observation = dict(zip(observation_windows, responses))
print experimental_observation

# <markdowncell>

# ### Enumeration machinery: enumerate all the possible colorings of windows and window unions
#     

# <markdowncell>

# #### First, a represenation of windows

# <codecell>

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

# <codecell>

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
    
##in need of a uniquify. got this one off of stackoverflow
def uniquify(seq): 
   # order preserving
   noDupes = []
   [noDupes.append(i) for i in seq if not noDupes.count(i)]
   return noDupes

# <codecell>

##==exercise!
win = window((2,3,4))
print win.tup
print win.string
print win.powerset().strings, len(win.powerset().strings)
print win.powerset().tups, len(win.powerset().tups)
foo = win.powerset(nonempty=True, strict = True).strings
print foo, len(foo)

# <codecell>

win = window((1,2,3,4))
cw = win.powerset()
print cw.tups
print cw.strings
print cw.reduce2window().string
print collection_of_windows({}).strings
print collection_of_windows([(2,)]).tups

# <markdowncell>

# #### Let's collect counting functions and put them here. Useful for later...

# <codecell>

##some counting functions
def count_atom_colorings(n,t,doprint=False):
    num = (2**n-1)**t
    if doprint:
        print 'length: %d' %(num)
        print 'GB: %f' %(num/(1024.**3))
    return num


##number of t non-empty n-bit strings that collectively encode exactly n colors
def count_atom_colorings_exactly_n(n,t):
    return int((2**n-1)**t-np.sum(map(lambda k: (2**(n-k)-1)**t*comb(n,k), range(1,n))))

# <markdowncell>

# #### This is the machinery for enumerating all of the possible cardinalities of windows and their unions

# <codecell>


##generate all possible t nonempty n-bit string segments
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
    
    
# def union_powerset(window_colorings,doprint=False):
#     '''given a set of window colorings (encoding as n-bit strings), return the cardinality of the union of the windows
#     in all subsets.'''
# #   indices = window(range(1,len(window_colorings)+1)).powerset(nonempty=True)
#     pwr = map(lambda idx: [window_colorings[ii-1] for ii in idx], indices.tups)
#     if doprint:
#         print 'indices: %s' %(indices.tups,)
#     unions = map(multi_union, pwr)
#     if doprint:
#         for p,u in zip(pwr,unions):
#             print '%s |---> %d' %(p,u)
#     return unions

# <codecell>

##bundle all of the above into a convenient (?) object
class consistent_cardinalities(object):
    '''
    given a set of t windows, enumerate all consistent cardinalities of the union powerset of the windows
    consistent_cardinalities(n,window_names, doprint=False,dotime=True)
    window_names will generally be lists of integers. 
    Returns a pandas data frame. Each column is a string 'i,j,k,...' specifying a union of the windows given in "window_names",
    which must a list or tuple of strings (each string must be a digit). Rows give the cardinality of each of the window unions.
    Strings are sorted.
    '''

    def __init__(self, n, window_object, redundant = False):
        self.t = len(window_object.tup)
        self.windows = window_object
        self.window_names = self.windows.powerset(nonempty=True)
        unions = []
        for win_clrs in generate_window_colorings(n,self.t,bits=True):
            uu = self.union_powerset(win_clrs, doprint=False)
            if redundant:
                unions += [uu]
            elif uu not in unions:
                unions += [uu]
        self.df = pd.DataFrame(np.array(unions), columns = self.window_names.strings)
    
    def union_powerset(self, window_colorings,doprint=False):
        '''given a set of window colorings (encoding as n-bit strings), return the cardinality of the union of the windows
        in all subsets.'''
        window_colorings = dict(zip(self.windows.tup, window_colorings))
        pwr = map(lambda idx: [window_colorings[ii] for ii in idx], self.window_names.tups)
        if doprint:
            print 'indices: %s' %(indices.tups,)
        unions = map(multi_union, pwr)
        if doprint:
            for p,u in zip(pwr,unions):
                print '%s |---> %d' %(p,u)
        return unions

# <codecell>

##===exercise!!
cur_wins = window((1,2))
win = 1
basis_windows = window((2,))
cc = consistent_cardinalities(size_of_universe, cur_wins, redundant=False)
cc.df


# <markdowncell>

# ### Weighting functions -- the functions (likelihoods, priors, etc.) that we marginalize

# <markdowncell>

#     * should be compatible with pd.DataFrame.apply()
#     * function for counting colorings
#     * a likelihood function
#    

# <markdowncell>

# #### Count colorings

# <codecell>

##for counting the number of colorings visible through a single target window, given the colorings visible through 
##the union of the target windows with some other target windows.

##measure the cardinality of each block of "Venn diagram" partition
def block_size(union_subset, basis_windows, union_values, size_of_universe,doprint=False):
    '''
    block_size(union_subset, basis_windows, union_values, size_of_universe,doprint=False)
    union_subset ~ a tuple of integers
    basis_windows ~ a windows object
    union_values ~ one row from a consistent_cardinalities.df object whose column names include the powerset of the basis windows
    size_of_universe ~ an integer
    '''
    
    adder = 0
    if len(union_subset) == len(basis_windows.tup):
        if doprint:
            print union_subset
        try:
            adder = size_of_universe - union_values[basis_windows.string]
        except KeyError:
            if not len(basis_windows.tup):
                adder = size_of_universe
            else:
                Exception('subset and basis are same size, but something is off')
        if doprint:
            print 'catch: adder = %d' %(adder)
        return adder
    for F in window(union_subset).powerset().tups: ##<<F is a tuple
        sgn = len(union_subset)-len(F)
        if doprint:
            print 'F |-------> %s' %(F,)
        
        ##take everyting in F out of the basis_windows, then powerset
        for T in window(filter(lambda tup: tup not in F, basis_windows.tup)).powerset().tups: ##<<T is a tuple
            if T:
                if doprint:
                    print 'T |--> %s' %(T,)
                adder += (-1)**(sgn+len(T)+1)*union_values[window(T).string]
                if doprint:
                    print 'adder: %d' %(adder)
    return adder

# <codecell>

##===exercise
cur_wins = window((0,1))
subset = window((0,)).tup
sou = 1
cc = consistent_cardinalities(sou, cur_wins, redundant=False)
row = cc.df.iloc[0]
print 'row: %s' %(row,)
block_size(subset,cur_wins,row,sou,doprint=True)

# <codecell>

##count the number of colorings of a single target window given that we know its unions with a bunch of other windows
def count_colorings(target_window, basis_windows, union_values, size_of_universe, doprint=False):
    '''
    count_colorings(target_window, basis_windows, union_values, size_of_universe, doprint=False)
    target_window ~ and integer
    basis_windows ~ a window object that does not contain the target window
    union_values ~ one row from a consistent_cardinalities.df object. column names include powerset of target+basis windows
    size_of_universe ~ an integer
    returns an integer that is the number of coloring of the target window given its cardinality the cardinalities of it's 
    unions with all the basis windows
    '''
    output = 1
    for S in basis_windows.powerset().tups: ##<<each S will be tuple
        if doprint:
            print 'S |-------------> %s' %(S,)
        upper = block_size(S, basis_windows, union_values, size_of_universe,doprint=False)
#         print upper        
        if doprint:
            print 'S |-------------> %s' %(S,)
        downer = block_size(S , window(basis_windows.tup+(target_window,)), union_values, size_of_universe,doprint=False)

#         print downer
        output *= comb(upper, downer)
#         if not output:
#             if doprint:
#                 print 'no maps with this configuration'
#                 return output
            
#         print output
    return output

# <codecell>

##===exercise!
cur_wins = window((1,33,4))
basis = window((1,4))
target_window = 33
cc = consistent_cardinalities(size_of_universe, cur_wins, redundant=False)
row = cc.df.iloc[0]
print row
print count_colorings(target_window, basis, row, size_of_universe, doprint=True)
print 'corner case: %d' %(count_colorings(target_window, window(()),row,size_of_universe))

# <codecell>

##===exercise!
cur_wins = window((2,3,4))
win = 2
basis_windows = window((3,4))
cc = consistent_cardinalities(size_of_universe, cur_wins, redundant=False)

def counting_func(row):
    return count_colorings(win, basis_windows, row[cc.window_names.strings], size_of_universe, doprint=False)
 
cc.df['count'] = cc.df.apply(counting_func,axis=1)
cc.df

# <markdowncell>

# #### Likelihood

# <codecell>

def counts(r,d,n):
    return np.array([comb(d,m)*comb(n-d, r-m) for m in range(int(min(r,d))+1)])

def lkhd(r,d,n,p_on,p_off):
    if r is None:
        Warning('Your response is NoneType. Returning lkhd=1')
        return 1.0
    probs = np.array([(1-p_on)**(d-m) * (p_on)**m * (p_off)**(r-m) * (1-p_off)**(n-d-r+m) for m in range(int(min(r,d))+1)])
    return counts(r,d,n).dot(probs)

# <codecell>

##===exercise!
n = 5
r = 3
p = (0.8, 0.17)
plt.plot(range(n+1), [lkhd(r,d,n, p[0],p[1]) for d in range(n+1)], label = '(%0.2f, %0.2f)' %(p))
p = (.6, 0.45)
plt.plot(range(n+1), [lkhd(r,d,n, p[0],p[1]) for d in range(n+1)], label = '(%0.2f, %0.2f)' %(p))
p = (0.53, 0.49)
plt.plot(range(n+1), [lkhd(r,d,n, p[0],p[1]) for d in range(n+1)], label = '(%0.2f, %0.2f)' %(p))

##pathological: p_on is less than 0.5
p = (0.44, 0.13)
plt.plot(range(n+1), [lkhd(r,d,n, p[0],p[1]) for d in range(n+1)], label = '(%0.2f, %0.2f)' %(p))

##pathological: p_off is greater than 0.5
p = (0.88, 0.74)
plt.plot(range(n+1), [lkhd(r,d,n, p[0],p[1]) for d in range(n+1)], label = '(%0.2f, %0.2f)' %(p))
plt.legend(loc = 'upper left', ncol = 2, prop = {'size': 8})
plt.xlabel('d')
plt.ylabel('likelihood r | d')

# <codecell>

##==complexity. assum p_on=p_off = 0.5. How does likelihood grow as n?
n = range(0,15)
r = 3
print [.5**num*comb(num,r) for num in n]
plt.plot(n, [.5**num*comb(num,r) for num in n])

# <markdowncell>

# ### Summing function

# <markdowncell>

# * A summing loop
#     - sums over all unit windows
#     - figures out the unit windows and window unions to manipulate
#     - enumerates consistent cardinalities of the window unions
#     - runs the sum of product operations over the weighting functions.
#     - constructs the message function for the next iteration

# <codecell>

print count_atom_colorings(size_of_universe, len(unit_windows.tup))
print count_atom_colorings_exactly_n(size_of_universe, len(unit_windows.tup))

# <codecell>

collect_messages[0]

# <codecell>

cc.df

# <codecell>

def marginalize(experimental_observation, size_of_universe, likelihood_kernel = lambda r,d,n: lkhd(r,d,n,p_on,p_off), doprint=False):

    ##initialize carryover windows = null set
    carryover_windows = collection_of_windows({})

    ##
    collect_messages = []


    ##assume that the experimental data comes as a dictionary of tuple:response pairs
    ##first, make sure all of the keys are sorted like they should be
    exp_obs = dict(zip( [ window(key).tup for key in experimental_observation.keys() ], experimental_observation.values()))
    if doprint:
        print exp_obs

    ##get all of the unit windows (unique integers) from the collection of windows in the experimental data
    unit_windows = collection_of_windows(exp_obs.keys()).reduce2window()
    if doprint:
        print unit_windows.string
        

    ##initialize message
    message = 1

    ##loop to count
    for win in unit_windows.tup:
        print '===============target window: %d' %(win)

        cur_obs = collection_of_windows([obs for obs in exp_obs.keys() if win in obs])  ## cur_obs = {(),(),...} each tuple is window-like
        if doprint:
            print 'current observations: %s' %(cur_obs.strings)

        ##package observered responses
        resp_dict = dict((co,exp_obs[co]) for co in cur_obs.tups)

        ##pop cur_obs
        [exp_obs.pop(co) for co in cur_obs.tups]

        ##get all the unit windows in these observations + plus any carried-over windows from last iteration
        ##NOTE: cur_wins.powerset() will be a superset of cur_obs
        cur_wins = collection_of_windows(uniquify(cur_obs.tups+carryover_windows.tups)).reduce2window()
        if doprint:
            print 'current windows: %s' %(cur_wins.string)


        ##define consistent cardinalities for cur_wins.powerset()
        cc = consistent_cardinalities(size_of_universe, cur_wins)
        
        try:
            cc.df = cc.df.merge(message,how='right', on=list(carryover_windows.strings))         
        except:
            ##assuming  constant message
            cc.df['message'] = message
            ##this should mean it's the first iteration
            if doprint:
                print 'window: %s, assuming constant or no message' %(win)



        ##of the powerset of current windows, split the summands (contains w) from the function arguments (does not contain w)
        summands = collection_of_windows([wu for wu in cc.window_names.tups if win in wu])
        arguments = collection_of_windows([wu for wu in cc.window_names.tups if win not in wu])
        basis_windows = window(filter(lambda w: w != win, cc.windows.tup))
        if doprint:
            print 'summands: %s' %(summands.strings)
            print 'arguments: %s' %(arguments.strings)
            print 'basis: %s' %(basis_windows.string)

        ##build the weighting functions
        def dummy_func(row):
            return 1

        def counting_func(row):
            return count_colorings(win, basis_windows, row[list(cc.window_names.strings)], size_of_universe)


        def likelihood_func(row):
            responses = [resp_dict[key] for key in cur_obs.tups] ##<<explicit order preservation!
            return np.prod(map(lambda r,d: likelihood_kernel(r,d,size_of_universe), responses, row[cur_obs.strings]))


        ##apply
        cc.df['likelihood'] = cc.df.apply(likelihood_func,axis=1)
        cc.df['prior'] = cc.df.apply(dummy_func,axis=1)
        cc.df['count'] = cc.df.apply(counting_func,axis=1)

        try:
            cc.df['product'] = cc.df[['likelihood','prior','count','message']].prod(axis=1)
        except KeyError:
            print 'no message to multiply'
            cc.df['product'] = cc.df[['likelihood','prior','count']].prod(axis=1)

        #sum to construct message, 
        try:
            message = cc.df.groupby(list(arguments.strings),as_index=False)['product'].sum() 
            message.rename(columns={'product':'message'},inplace=True)
            collect_messages += [message]
            carryover_windows = arguments
        except ValueError:
            print 'window %s, no groupby. summing' %(win)
            message = cc.df['product'].sum()
            collect_messages += [message]

    print '*******final message is: %f' %(message)
    return message, collect_messages, cc    

# <codecell>

##===exercise!!
start = time()
p_on,p_off = .99, .01
final_message, cm, cc = marginalize(experimental_observation, size_of_universe)
print 'elapsed time: %f' %(time()-start)

# <markdowncell>

# ### Test: using fake data (with random response) infer noise params. Should get 0.5/0.5. 

# <codecell>

##===try to infer parameters of the random data
prob_on,prob_off = noise_grid(5,5)
lkhds = []
for p_on,p_off in zip(prob_on, prob_off):
    lk, _ , _ = marginalize(experimental_observation, size_of_universe)
    lkhds += [lk]
 

# <codecell>

fig = plt.figure()
ax = fig.add_subplot(1,1,1, aspect=1.0, axisbg=np.array([84, 149, 178] )/400.)#13,76,85])/350., ) #'gray')##[13,76,85]
cmap2 = plt.cm.get_cmap('gist_heat', 30)
mappable = ax.scatter(prob_on,prob_off,s=1500,c=lkhds, cmap=cmap2, marker='o',linewidth=0,alpha=.9)
plt.colorbar(mappable)

# <codecell>

marg_dict = {'prob_on':prob_on, 'prob_off':prob_off, 'lkhds':np.log(lkhds+eps)}

# <codecell>

marg_pd = pd.DataFrame(marg_dict)
marg_pd

# <codecell>

marg_pd.sort(columns=['lkhds'])

# <codecell>

##^^^looks good

# <markdowncell>

# ###Create some better fake data using an actual object map.

# <markdowncell>

# * generate responses from map with known noise params
# * test lkhd function against histogram for a few windows, to make sure everything works
# * marginalize, recover p_on, p_off, and n.

# <codecell>

##create a fake map
fake_map = spm(9,9,500,cluster_pref = 'random',number_of_clusters=3)

# <codecell>

map_img = fake_map.nn_interpolation()
map_img = Image.fromarray(np.squeeze(map_img)).resize((600,370))

# <codecell>

plt.imshow(map_img)

# <codecell>

##read in probes from the experiment
probe_exp = pd.load('/media/tnaselar/Data/scratch/z1_KL.pkl')

# <codecell>

probe_exp.head()

# <codecell>

probe_dict = pd.load('/musc.repo/Data/tnaselar/imagery_psychophysics/multi_poly_probes/probes/candle_01_letterbox_img__probe_dict.pkl')

# <codecell>

plt.imshow(probe_dict['mask'][0])
print probe_dict['index'][0]

# <codecell>

probe = Image.fromarray(probe_dict['mask'][0].astype('uint8')*255,mode='L')

# <codecell>

plt.imshow(probe)

# <codecell>

masked_map = apply_mask(map_img,probe,0)

# <codecell>

plt.imshow(masked_map)

# <codecell>

def window_response(map_image, window_mask, size_of_universe, p_on=1, p_off=0):
    objects_in_window = len(apply_mask(map_image, window_mask, 0).getcolors())-1 ##the -1 is for 0
    on_resp=binomial(objects_in_window,p_on)
    off_resp = binomial(size_of_universe-objects_in_window,p_off)
    return on_resp+off_resp
    

# <codecell>

size_of_universe = 7
deterministic_resp = window_response(map_img, probe, size_of_universe)
p_on, p_off = .85, .1
noisy_resp = [window_response(map_img, probe, size_of_universe,p_on=p_on,p_off=p_off) for _ in range(1000)]

# <codecell>

plt.hist(noisy_resp,normed=True,bins=size_of_universe+1,range=(0,7))
d = deterministic_resp
n = size_of_universe
plt.plot(range(size_of_universe+1), [lkhd(r,d,n, p_on, p_off) for r in range(size_of_universe+1)])

# <codecell>


