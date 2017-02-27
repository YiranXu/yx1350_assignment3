from itertools import permutations
import numpy as np
def zbits(n, k):
    #'''
    #prints all binary strings of length n that contain k zero bits, one per line
    #'''
    #create a list with k zero bit, (n-k) one bit
    initial=['0']*k+['1']*(n-k)
    item_list=list()
    #use permutations to find all all binary strings of length n that contain k zero bits
    #notice: there are a lot of repetitions because the repetition of one and zero
    #store each permutations to another list in order to select unique ones later
    for item in permutations(initial,n):
        item_list.append(''.join(item))
    #select unique permutation of length n that contain k zero bits
    results=np.unique(item_list)
    #conver the results to a set so that the order within results would not matter
    return set(results) 
