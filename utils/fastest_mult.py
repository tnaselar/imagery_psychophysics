##what is fastest way to do many matrix multiplications?

import numpy as np
from scoop import futures
import time


maps = np.random.random((7770, 32,32))
masks = np.random.randint(0,2,size=(11**2,32,32))


if __name__ == '__main__':
  

  #start = time.time()
  #biz = [list(map(lambda q: np.nonzero(np.unique(q))[0].size, f*maps)) for f in masks]
  #print 'serial map + list. comp took %f seconds' %(time.time()-start) 
  
  #start = time.time()
  #bang = [futures.map(lambda q: np.nonzero(np.unique(q))[0].size, f*maps) for f in masks]
  #print "par'l. map + list. comp took %f seconds" %(time.time()-start)
  
  start = time.time()
  futures.map(lambda x: [np.nonzero(np.unique(q))[0].size for q in x*masks], maps)
  print "par'l. map + list. comp took %f seconds" %(time.time()-start)
 
  start = time.time()
  map(lambda x: [np.nonzero(np.unique(q))[0].size for q in x*masks], maps)
  print "ser'l. map + list. comp took %f seconds" %(time.time()-start)

 
 
  #start = time.time()
  #bop = [[np.nonzero(np.unique(q))[0].size for q in f*maps] for f in masks]
  #print 'pure list. comp took %f seconds' %(time.time()-start)
  
  #start = time.time()
  #blarg = []
  #for f in masks:
    #for q in f*maps:
      #blarg.append(np.nonzero(np.unique(q))[0].size)
  #print 'pure for-loop took %f seconds' %(time.time()-start)
  
  