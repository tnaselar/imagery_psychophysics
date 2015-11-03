##poisson binomial distribution

n = np.arange(1,6)  ##list of possible objects ( = 5)
r = 3  ##response

i,j,k = np.meshgrid(n,n,n)
num_poss = i.size

all_pairs_rep = np.hstack([i.reshape((num_poss,1)),j.reshape((num_poss,1)),k.reshape((num_poss,1))])

def find_reps_in_stack(stack):
	num_cols = stack.shape[1]
	return np.array(map(lambda x: len(unique(x))==num_cols, stack)).flatten()

all_subsets_of_length_n = all_pairs_rep[find_reps_in_stack(all_pairs_rep),:]
	