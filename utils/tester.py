class foo(object):
    def __init__(self, x):
        self.x = x
   
    def compute(self,z,y):
        return self.x*z*y
    
class baz(object):
  def __init__(self, y):
    self.y = y
    
class bop(foo, baz):
  def __init__(self,x,y):
    super(bop,self).__init__(x)
    super(bop,self).__init__(y)

class flop(object):
  def __init__(self, x):
    self.x = x
  def compute(self,q):
    return self.x*q