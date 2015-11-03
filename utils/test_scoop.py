##test scoop
import numpy as np
from scoop import futures
import time
from imagery_psychophysics.utils.tester import flop

dim1 = 64
dim2 = 64
dim3 = 10
rep = 1000
x = np.random.random((dim1,dim2))

y = [np.random.random((dim3,dim1,dim2))]*rep
z = np.random.random((rep,dim3,dim1,dim2))


if __name__ == '__main__':
  
  start = time.time()
  for q in y:
    x*q
  print 'for loop took %f seconds' %(time.time()-start) 
  
  start = time.time()
  [x*q for q in y]
  print 'list comp took %f seconds' %(time.time()-start)
  
  start = time.time()
  x*z
  print 'broadcasting took %f seconds' %(time.time()-start) 
  

  start = time.time()
  map(lambda q: x*q, y)
  print 'serial map took %f seconds' %(time.time()-start) 

  
  
  start = time.time()
  futures.map(lambda q: x*q, y)
  print 'parallel map took %f seconds' %(time.time()-start)
  
      
  flong = flop(x)
  start = time.time()
  futures.map(flong.compute, y)
  print 'parallel map with method took %f seconds' %(time.time()-start)
  
