import numpy as np
from mpi4py import MPI
comm=MPI.COMM_WORLD
rank=comm.Get_rank() #get rank for this process
size=comm.Get_size() #get total number of processes

num=np.zeros(1)
if rank==0: 
#Process 0 reads a value from the user and verifies that it is an integer less than 100.
	input=input("Please enter an integer less than 100: \n")
	#verify input and catch exception
	try: 
		num[0]=int(input)
			
	except ValueError:
		print("You did not enter an integer. ")
	else:
		try:
			if num[0]>=100:
				raise ValueError
		except ValueError:
			print ("The interger you entered is greater than 100. Please try again.")
		
		else: 
		#send the input number to process 1
			if size>1:
				#print("before send rank0",num[0])
				comm.Send(num,dest=1)
				comm.Recv(num,source=size-1)
				print("input is",num[0])
			else: #only one process
				print("size is 0, value is",num[0])
if rank!=0 and rank!=(size-1):
#for process i (except process 0 and the last process),
#process i receives a value from process i-1, muplitiplies it by its rank,
#and sends the value to process i+1 which multiplies it by i+1
	comm.Recv(num,source=rank-1) #receives a value from process i-1
	num*=rank 
	comm.Send(num,dest=rank+1) #send to process i+1 			
if rank==size-1:
	if rank!=0:
		comm.Recv(num,source=rank-1)
		num*=rank
		comm.Send(num,dest=0)
#print("rank,size,value",rank,size,num[0])
