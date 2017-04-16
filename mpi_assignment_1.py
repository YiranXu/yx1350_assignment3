'''
This MPI program prints "Hello from process i" if rank is even and
prints "Goodbye from process i" if rank is odd.
'''

from mpi4py import MPI
comm=MPI.COMM_WORLD
rank=comm.Get_rank() #get rank for this process

if rank%2==0: #rank is even
	print("Hello from process "+str(rank))
else: #rank is odd
	print("Goodbye from process "+str(rank))
