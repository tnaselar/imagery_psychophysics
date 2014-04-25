
##some useful arithmetic operations for EM
def apply_to_col_pairs(X, Y, function):
  import numpy as np
  '''z = apply_to_col_pairs(V_by_K, V_by_M, function)
  apply 'function' to each pair of columns in two 2D arrays that have the same number of rows
  "function' should be defined for a matrix X and an array Y. So any broadcastable operation will work 
  z ~ V_by_K_by_M
  '''
  V = X.shape[0]
  M = Y.shape[1]
  z = np.dstack([function(X,Y[:,ii].reshape((V,1))) for ii in range(M)])
  return z
  

def log_prod_sum(X,Y):
  import numpy as np
  '''log_prod_pow(V_by_K, V_by_M)
     z = sum(y*log(x)) ~ M_by_K
  '''
  def log_prod(X,Y):
    return np.log(X)*Y
     
  return np.squeeze(apply_to_col_pairs(X,Y,log_prod).sum(axis=0)).T


##some stuff for viewing 
def get_colors_from_colormap(cmap = 'Paired', ncols = 8 ):
  import matplotlib.pyplot as plt
  import matplotlib.colors as colors
  import matplotlib.cm as cmx
  import numpy as np
  values = range(ncols)
  jet = cm = plt.get_cmap(cmap) 
  cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
  scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
  colors = np.zeros((ncols,4)) ## << RGBA
  for idx in range(ncols):
    colors[idx,:] = scalarMap.to_rgba(values[idx])
  return colors
    
def plain_plot(ax):
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xlabel('')
  ax.set_ylabel('')
  
#def view_lkhd_dist_as_images(dist):
  #from numpy import sqrt
  #from imagery_psychophysics.utils.montage import montage 
  #V,K = dist.shape
  #dist = dist.reshape((sqrt(V),sqrt(V),K))
  #montage(dist)
  
#def view_MAP_as_image(post_dist):
  #from numpy import max, sqrt
  #from matplotlib.pyplot import imshow, figure
  #V = post_dist.shape[0]
  
  #figure('MAP object for each pixel')
  #imshow(post_dist.argmax(axis=1).reshape((sqrt(V), sqrt(V))))
  
##define a model class
class mix_hog(object):
  '''
  mix_hog(prior_dist, lkhd_dist)
  
  prior_dist ~ 1 x K, K = number_of_states
  lkhd_dist ~ V x K, V = vocab_size
  
  '''
  def __init__(self, prior_dist, lkhd_dist):
    from numpy import finfo,double
    self.prior_dist = prior_dist
    self.lkhd_dist = lkhd_dist
    self.vocab_size, self.number_of_states = lkhd_dist.shape
    self.underflow_k = 10  		 ##to prevent numerical underflow
    self.too_small = finfo(double).tiny ##to prevent log(0) errors
    self.smoothing_param = 0		 ##controls smoothing of lkhd_dist / prevents log(0) errors
    self.smooth_lkhd_dist() ## << in case any of the probs = 0
    
  def log_likelihood(self, document):

    '''
       log_likelihood(documents)
       documents ~ V x M matrix of counts for each of the V words
       
       
       returns approx. log_likelihood (scalar) of the data:
       
       z = sum ( log ( sum (prior * prod(lkhd**documents) ) ) )
       
       the first sum is over documents, the second sum is over states, and the product is over vocabulary words
       
       this is numerically stable. for algorithm see
       http://u.cs.biu.ac.il/~shey/919-2011/index.files/underflow%20and%20smoothing%20in%20EM.pdf 
    
    '''
    import numpy as np
    import pdb
    M = document.shape[1]    
    z = np.log(self.prior_dist)+log_prod_sum(self.lkhd_dist,document) ## M x K 
    m = np.atleast_2d(np.max(z, axis=1)).reshape((M,1)) ## M x 1
    dx = np.array(z-m) >= -self.underflow_k ## M x K
    lkhd_part = np.atleast_2d(np.log(np.array(np.exp(z-m)*dx).sum(axis=1))).reshape((M,1)) ##M x 1
    return np.array(m+lkhd_part).sum(axis=0)
    
  def print_dimensions(self):
    print 'number of states (K) : %d' %(self.number_of_states)
    print 'vocab size (V) : %d' %(self.vocab_size)
    print 'prior array: %d by %d' %self.prior_dist.shape
    print 'lkhd matrix: %d by %d' %self.lkhd_dist.shape
    print 'document matrix should be (V) by (number_of_documents)'
    print 'the i-th, j-th element in doc. matrix is the # occurences of word i in document j'

  
  def smooth_lkhd_dist(self):
    self.lkhd_dist[self.lkhd_dist < self.too_small] = self.too_small
    self.lkhd_dist = self.lkhd_dist/self.lkhd_dist.sum(axis=0, keepdims=True)
  
  
  def show_params(self):
    from matplotlib.pyplot import imshow, figure, close
   
    figure('prior')
    imshow(self.prior_dist, cmap='gray')
    
    figure('likelihood distribution')
    imshow(self.lkhd_dist)
  
  ##should return M x K  
  def posterior(self, document):
    '''
    w = posterior(documents) ~ M x K
    documents ~ V x M
    posterior over hidden states given for each document.
    numerically stable approximation.
    
    '''
    import numpy as np
    M = document.shape[1]
    z = np.log(self.prior_dist)+log_prod_sum(self.lkhd_dist,document)
    m = np.atleast_2d(np.max(z, axis=1)).reshape((M,1)) ## M x 1
    dx = np.array(z-m) >= -self.underflow_k ## M x K
    denominators = np.atleast_2d(np.array(np.exp(z-m)*dx).sum(axis=1)).reshape((M,1)) ##M x 1
    return np.exp(z-m)/denominators*dx
      
  
  def lkhd_m_step(self, documents):
    '''update the likelihood distribution
       documents ~ V x M
       smoothing_param = 0
       lkhd_dist = lkhd_m_step(documents, [smoothing_param]) ~ V x K
    '''
    import numpy as np
    r = self.posterior(documents) ## M x K
    beta = documents.dot(r) ## V x K
    T = np.atleast_2d(beta.sum(axis=0)) ##1 x K 
    return (beta+self.smoothing_param)/(T+self.vocab_size*self.smoothing_param)  ##(V x K) / (1 x K) ~ V x K
  
  def prior_m_step(self, documents):
    '''update the prior distribution
       documents ~ V x M
       prior_dist = prior_m_step(documents) ~ 1 x K
    '''
    import numpy as np
    M = documents.shape[1]
    pies = np.atleast_2d(self.posterior(documents).sum(axis=0))/M
    pies[pies < self.too_small] = self.smoothing_param
    pies = pies/pies.sum() ##
    return pies 
    
  def run_em(self, documents, stop_delta = 10**-4):
    '''update the lkhd_dist and prior_dist parameters given a set of documents
       documents ~ V x K
       run_em(documents)
    
    '''
    import numpy as np
    import pdb
    lkhd_delta = stop_delta*100
    old_data_lkhd = self.log_likelihood(documents)
    print '===%0.5f' %(old_data_lkhd)
    ## note that the likelihood can only improve, so we don't need to be conservative and keep old solutions at hand
    while lkhd_delta > stop_delta:
      #pdb.set_trace()
      new_lkhd_dist = self.lkhd_m_step(documents)
      new_prior_dist = self.prior_m_step(documents)
      self.prior_dist = new_prior_dist
      self.lkhd_dist = new_lkhd_dist
      new_data_lkhd = self.log_likelihood(documents)
      lkhd_delta = new_data_lkhd - old_data_lkhd
      print '===%0.5f' %(new_data_lkhd)
      old_data_lkhd = new_data_lkhd

      
      
      
class mix_hog_image(mix_hog):
  '''assumes the mix_hog model is describing objects in images.
     adds some viewing methods for visualizing the parameters of the model as images
  '''

  def __init__(self, prior_dist, lkhd_dist, colormap = 'Spectral'):
    super(mix_hog_image, self).__init__( prior_dist, lkhd_dist )
    self.colormap = get_colors_from_colormap(cmap = colormap, ncols = self.number_of_states )
  
  def plain_plot(self,ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
  
  def view_lkhd_dist(self):
    from numpy import sqrt, arange, ones
    import matplotlib.pyplot as plt
    N = tuple([sqrt(self.vocab_size)])*2
    K = self.number_of_states
    f = plt.figure()
      
    for ii in arange(K)+1:
      plt.subplot(1,K,ii)
      ax = plt.gca()
      plain_plot(ax)
      ax.set_title('obj. %d' %(ii), color=self.colormap[ii-1,:], backgroundcolor = ones(4))
      plt.imshow(self.lkhd_dist[:,ii-1].reshape(N), cmap='gray')
      
  
  def view_MAP(self):
    from numpy import max, sqrt, eye
    from matplotlib.pyplot import imshow, figure, gca
    from matplotlib.colors import ListedColormap
    post_dist = self.posterior(eye(self.vocab_size))
    N = tuple([sqrt(self.vocab_size)])*2
    figure('MAP object for each pixel')
    plain_plot(gca())
    imshow(post_dist.argmax(axis=1).reshape((N)), cmap=ListedColormap(self.colormap))
  
  def view_prior(self):
    from matplotlib.pyplot import bar,figure, gca
    from numpy import arange,squeeze 
    figure('Prior distribution')
    ax = gca()
    ax.xaxis.set_ticks(arange(self.number_of_states)+1)
    ax.xaxis.set_label('object id')
    bar(arange(self.number_of_states)+1, squeeze(self.prior_dist), color = self.colormap, align='center')
    
     
   