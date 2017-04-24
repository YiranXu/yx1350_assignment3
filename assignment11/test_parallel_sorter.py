#this test uses mpi4py to test funcions in parallel_sorter.py

import parallel_sorter
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#test scatter_data(data,size,rank)
#this test works for 2 processes
data=parallel_sorter.scatter_data(np.array([7,13,11]),size,rank)
if size==2:
#range for [7,13,11] is 6. 
#With two processors, the first would get numbers from 7 to 10,the second would get numbers from 11 to 13.
	if rank==0:
		np.testing.assert_array_equal(data,[7])
	if rank==1:
		np.testing.assert_array_equal( data,[13,11])

#test whether the final array is indeed sorted
if rank==0:
#print("sort",rank,parallel_sorter.sort_all(data))
	np.testing.assert_array_equal(parallel_sorter.sort_all(data),[7,11,13])
else:
	assert parallel_sorter.sort_all(data) is None
