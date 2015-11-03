##measure the cardinality of each block
def block_size(union_subset, union_value_dict):
    '''notation guide:
    union_subset = S
    union_value_dict[T] = d_T
    '''
 
    basis_set_indices = integers_from_keys(union_value_dict)
 
    adder = 0
    if len(union_subset) == len(basis_set_indices):
        return adder
    for F in powerset(union_subset):
        print 'F |-------> %s' %(F,)
        sgn = (-1)**(len(S)-len(F))
        for T in powerset(set(basis_set_indices)-set(F)):
            if T:
                print 'T |--> %s' %(T,)
                adder += (-1)**(len(T)+1)*union_value_dict[T]*sgn
#                 print 'adder: %d' %(adder)
    return adder









def count_sets(target_index, union_value_dict):
    basis_set_indices = list(set(integers_from_keys(union_value_dict)) - set((target_index,)))  ##these are the basis sets
    basis_values_only = ignore_keys((target_index,), union_value_dict)
    output = 1
    for S in powerset(basis_set_indices):
	print 'S |-------------> %s' %(S,)
	upper = block_size(S, basis_values_only)
	print 'S |-------------> %s' %(S,)
	downer = block_size(S , union_value_dict)
	print upper
	print downer
	output *= comb(upper, downer)
	print output
    return output




##integrate over all possible cardinalities of partition blocks 
number_of_objects = 4
number_of_basis_sets = 2 ##<<should include the target set
target_index = 0
target_cardinality = 3
union_value_dict = {}
all_possible_values_of_unions = list(cwr(range(1,number_of_objects+1), number_of_basis_sets))
all_possible_unions_of_sets = allsubsets(number_of_basis_sets)
all_counts = [{}]*len(all_possible_values_of_unions)
for ii,cc in enumerate(all_possible_values_of_unions):
    for ci in all_possible_unions_of_sets:
        all_counts[ii][ci] = cc[ii]
print all_counts




{
 "metadata": {
  "name": "",
  "signature": "sha256:75d94f811fff938e8318a7479ed830d7eb8426bfdcf9ac78cf25ac410e2b107d"
 },
 "nbformat": 3,
 "nbformat_minor": 0,
 "worksheets": [
  {
   "cells": [
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "import numpy as np\n",
      "from itertools import combinations, chain\n",
      "from itertools import combinations_with_replacement as cwr\n",
      "from math import factorial as bang\n",
      "from scipy.misc import comb"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 1
    },
    {
     "cell_type": "heading",
     "level": 4,
     "metadata": {},
     "source": [
      "helpful functions"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "##powerset of a list\n",
      "def powerset(iterable):\n",
      "    s = list(iterable)\n",
      "    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))\n",
      "\n",
      "##all subsets of the integers -- redundant.\n",
      "allsubsets = lambda n: list(chain(*[combinations(range(n), ni) for ni in range(n+1)]))\n",
      "\n",
      "##unique integers from keys\n",
      "def integers_from_keys(mydict):\n",
      "    return list(np.unique(sum(mydict.keys(),())))  ## == [m]\n",
      "\n",
      "##return subset of a dictionary with certain keys ignored\n",
      "def ignore_keys(ignore_any_key_with, mydict):\n",
      "    return {key: mydict[key] for key in mydict.keys() if all([x not in key for x in ignore_any_key_with])}\n",
      "    "
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 2
    },
    {
     "cell_type": "heading",
     "level": 4,
     "metadata": {},
     "source": [
      "construct an intersect-complement-union partion using indexed basis sets "
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "##measure the cardinality of each block\n",
      "def block_size(union_subset, union_value_dict):\n",
      "    '''notation guide:\n",
      "    union_subset = S\n",
      "    union_value_dict[T] = d_T\n",
      "    '''\n",
      " \n",
      "    basis_set_indices = integers_from_keys(union_value_dict)\n",
      " \n",
      "    adder = 0\n",
      "    if len(union_subset) == len(basis_set_indices):\n",
      "        return adder\n",
      "    for F in powerset(union_subset):\n",
      "        print 'F |-------> %s' %(F,)\n",
      "        sgn = (-1)**(len(S)-len(F))\n",
      "        for T in powerset(set(basis_set_indices)-set(F)):\n",
      "            if T:\n",
      "                print 'T |--> %s' %(T,)\n",
      "                adder += (-1)**(len(T)+1)*union_value_dict[T]*sgn\n",
      "#                 print 'adder: %d' %(adder)\n",
      "    return adder\n"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 3
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "number_of_basis_sets = 3\n",
      "S = (1,0)\n",
      "union_value_dict = {}\n",
      "for ci in allsubsets(number_of_basis_sets):\n",
      "    union_value_dict[ci] = 2\n",
      "block_size(S, union_value_dict)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "F |-------> ()\n",
        "T |--> (0,)\n",
        "T |--> (1,)\n",
        "T |--> (2,)\n",
        "T |--> (0, 1)\n",
        "T |--> (0, 2)\n",
        "T |--> (1, 2)\n",
        "T |--> (0, 1, 2)\n",
        "F |-------> (1,)\n",
        "T |--> (0,)\n",
        "T |--> (2,)\n",
        "T |--> (0, 2)\n",
        "F |-------> (0,)\n",
        "T |--> (1,)\n",
        "T |--> (2,)\n",
        "T |--> (1, 2)\n",
        "F |-------> (1, 0)\n",
        "T |--> (2,)\n"
       ]
      },
      {
       "metadata": {},
       "output_type": "pyout",
       "prompt_number": 4,
       "text": [
        "0"
       ]
      }
     ],
     "prompt_number": 4
    },
    {
     "cell_type": "heading",
     "level": 4,
     "metadata": {},
     "source": [
      "count the number of sets possible by building a set from the blocks of a partition. For each block of a partition, we intersect the target set with the blocks of the partition. If the block contains n elements, and the intersection of the target with the block contains k elements, then there are n-choose-k ways of constructing that little compartment of the target set."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "\n",
      "def count_sets(target_index, union_value_dict):\n",
      "    basis_set_indices = list(set(integers_from_keys(union_value_dict)) - set((target_index,)))  ##these are the basis sets\n",
      "    basis_values_only = ignore_keys((target_index,), union_value_dict)\n",
      "    output = 1\n",
      "    for S in powerset(basis_set_indices):\n",
      "        print 'S |-------------> %s' %(S,)\n",
      "        upper = block_size(S, basis_values_only)\n",
      "        print 'S |-------------> %s' %(S,)\n",
      "        downer = block_size(S , union_value_dict)\n",
      "        print upper\n",
      "        print downer\n",
      "        output *= comb(upper, downer)\n",
      "        print output\n",
      "    return output"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 13
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "number_of_objects = 4\n",
      "number_of_basis_sets = 2\n",
      "union_value_dict = {}\n",
      "for ci in allsubsets(number_of_basis_sets):\n",
      "    union_value_dict[ci] = 2\n",
      "union_value_dict[(number_of_basis_sets-1,)] = number_of_objects\n",
      "\n",
      "print union_value_dict \n",
      "target_index = 0\n",
      "count_sets(target_index, union_value_dict)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "{(0, 1): 2, (): 2, (1,): 4, (0,): 2}\n",
        "S |-------------> ()\n",
        "F |-------> ()\n",
        "T |--> (1,)\n",
        "S |-------------> ()\n",
        "F |-------> ()\n",
        "T |--> (0,)\n",
        "T |--> (1,)\n",
        "T |--> (0, 1)\n",
        "4\n",
        "4\n",
        "1.0\n",
        "S |-------------> (1,)\n",
        "S |-------------> (1,)\n",
        "F |-------> ()\n",
        "T |--> (0,)\n",
        "T |--> (1,)\n",
        "T |--> (0, 1)\n",
        "F |-------> (1,)\n",
        "T |--> (0,)\n",
        "0\n",
        "2\n",
        "0.0\n"
       ]
      },
      {
       "metadata": {},
       "output_type": "pyout",
       "prompt_number": 16,
       "text": [
        "0.0"
       ]
      }
     ],
     "prompt_number": 16
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [],
     "language": "python",
     "metadata": {},
     "outputs": []
    }
   ],
   "metadata": {}
  }
 ]
}