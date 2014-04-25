import numpy as np
from imagery_psychophysics.src.mixture_of_histograms import mix_hog

V = 4096
K = 8
M = 1000

P = 47 ##pixels per probe

prior_dist = np.random.dirichlet(np.ones(K)).reshape((1,K))
lkhd_dist = np.random.dirichlet(np.ones(V), size=K).T

docs = np.zeros((V,M))
for ii in range(M):
  dx=np.random.choice(np.arange(V), size=P, replace=False)
  docs[dx,ii] = 1
  
mh = mix_hog(prior_dist, lkhd_dist)

mh.print_dimensions()

print mh.posterior(docs).shape
print mh.log_likelihood(docs)

mh.show_params()

