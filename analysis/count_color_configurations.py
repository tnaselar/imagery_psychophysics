from scipy.misc import comb
from math import factorial as bang
import numpy as np

##generate all t-length sets of non-empty binary strings of length n and < n bits sets to 1.
n = 2
t = 2

def generate_binary(n):

  # 2^(n-1)  2^n - 1 inclusive
  bin_arr = range(0, int(math.pow(2,n)))
  bin_arr = [bin(i)[2:] for i in bin_arr]

  # Prepending 0's to binary strings
  max_len = len(max(bin_arr, key=len))
  bin_arr = [i.zfill(max_len) for i in bin_arr]

  return bin_arr


base_list = generate_binary(n-1)
base_list.remove('0'*(n-1))
base_list = list(itertools.product(base_list, repeat=t))

##now add a 0 to each binary string in n positions
all_seqs = []
for kk in range(n):
	for seqs in base_list:
		all_seqs.append(tuple(map(lambda y: y[:kk]+'0'+y[kk:], seqs)))

atleastn = (2**n-1)**t
print 'at least n: %d' %(atleastn)
atmostn = len(set(all_seqs))
print 'at most n: %d' %(atmostn)


print np.sum(map(lambda k: (2**(n-k)-1)**t*comb(n,k), range(1,n)))

#counts = collections.Counter(all_seqs)

##how many terms will contain '00010

# adder = 0
# for k in range(2,n):
# 	for d in range(2,k+1):
# 		adder += comb(n, k)*comb(k,d)*(2**(n-d)- 1 - (d!=(n-1)))*(d-1)
# print adder	
# 
# calc =  ((2**(n-1)-1)**t)
# 
# 	
# print calc	
# 
# 
# counts = collections.Counter(all_seqs)	
# [k for k in counts.keys() if '00010'==k[0] if counts[k] == 4]
('00010', '10001')
('00010', '10011')