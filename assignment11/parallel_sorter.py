from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#rank 0 is root process where we get data, split and scatter to all processors
def scatter_data(data,size,rank):
	if rank == 0:
		range_of_data=np.ptp(data) #range of data
		number=int(range_of_data/size)+1 #how many elements in one process
		output=[]
		#split data to n processes(n=size) based on range of data
		for i in range(0,size):
			if i==0:
				min=np.amin(data)
				max=min+number-1
			else:
				min=max+1
				max=min+number-1
			#find elements in the range for rank i and put them in output[i] for process i
			output.append(data[np.where(np.logical_and(data>=min, data<=max))])
		#convert to numpy array for scattering
		data =np.asarray(output)
	else:
		data = None
	#scateer data to all processes
	data = comm.scatter(data, root=0)
	return data

def sort_all(data):
#this function sort data in every processor and gather them into root
	data=np.sort(data)
	#gather processed data to root
	data=comm.gather(data,root=0)
	if rank==0:
        #combine all arrays into one array
                data=np.concatenate(data)
	return data

if __name__=="__main__":
	data=None
	if rank==0:
		#input=input("Please enter an integer as length of data: \n")
		
		#randomly generate 10,000 unsorted data set
		data=np.random.randint(0,100000,10000)
	data=scatter_data(data,size,rank)
	data=sort_all(data)
	if rank==0:
		print(data)
