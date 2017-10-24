import sys
import datetime
import math
import os
import random
import numpy as np
from mpi4py import MPI

def binary_search(array,i,j,value):
	"""
	----------------------------------------------------------------------------
	Performing a binary search for the biggest item smaller then value on array
	---------------------------------------------------------------------------
	Preconditions:
		array - a list of data
		i - start index for the search
		j - end index for the search
		value - the item to find
	Postconditions:
		returning the index of the closest item to value, smaller or equal
	----------------------------------------------------------------------------
	"""
	# Stop condition for when the range is up to 2 values
	if (j-i<=1):
		# If the value is lower than the minimum value, return the index before it
		if (value<array[i]):
			return i-1;
		# If the value is higher than the small value but lower than the higher
		elif(value<=array[j]):
			return i;
		# The value is higher than the two indexes
		else:
			return j;
	# Comparing to the item found in the middle of the range
	k=i+(j-i)/2
	if (value<=array[k]):
		# Recursion call for the lower half of the range
		return binary_search(array,i,k,value)
	else:
		# Recursion call for the higher half of the range
		return binary_search(array,k+1,j,value)

def get_input(rank,comm):
	"""
	-----------------------------------------------------
	Process 0 is asking for number of items from the user
	and sends the generated lists to the other processes
	-----------------------------------------------------
	Preconditions:
		rank - the id of the current processsor
		comm - MPI communicator instance
	Postconditinos:
		Every process returns:
			n - number of items in lists a and b
			k - log of n in base 2
			r - n/k
			a - sorted list of random items in size n
			b - sorted list of random items in size n
	--------------------------------------------------------
	"""
	# Only rank 0 read input and generates the lists
	if (rank==0):
		print("Please enter the power of 2 representing the number of items to generate")
		k=input()
		LIM=3999999999
		n = 2**k
		a = generate_randoms(n,LIM)
		b = generate_randoms(n,LIM)
		#n=8
		#a=[12, 18, 23, 23, 33, 34, 43, 44]
		#b=[1, 10, 19, 20, 27, 31, 36, 44]
		#k=3

		# If n/k has a reminder, add a partition
		r = int(n/k)
		if (n%k>0):
			r+=1
	else:
		k = None
		a = None
		b = None
		n = None
		r = None
	
	return n,k,r,a,b
	
def generate_randoms(n,lim):
        """
        -------------------------------------------------------------
        Generating a list of n random numbers between 0 and lim
        -------------------------------------------------------------
        Preconditions:
                n - number of items to generate
                lim - highest value to generate
        Postconditions:
                returns a sorted list of n random items between 0-lim
        -------------------------------------------------------------
        """
        a=np.empty(n,dtype=np.uint32)
        increase=lim/n
        last_value=1
        for i in range(n):
                a[i] = random.randint(1,increase) + last_value
		last_value=a[i]
        return a

def merge(a,b,a_start,a_end,b_start,b_end):
	"""
	-------------------------------------------------------------------------
	Performing a merge of two local sorted lists
	-------------------------------------------------------------------------
	Preconditions:
		a,b - two sorted lists to be merged
		a_start,b_start - the indexes of a and b to start merging from
		a_end, b_end - the indexes of a and b to stop merging at
	Postconditions:
		merged and sortedlist c in size (a_end-a_start) + (b_end-b_start)
	-------------------------------------------------------------------------
	"""
	c = []
	# As long as we haven't reached to the end of the range for either of the lists
	# we continue to merge the smallest item available
	while (a_end>=a_start and b_end>=b_start):
		if (a[a_start]<=b[b_start]):
			c.append(a[a_start])
			a_start+=1
		else:
			c.append(b[b_start])
			b_start+=1

	# Merging the tail of list a when list b is already merged
	while (a_end>=a_start):
		c.append(a[a_start])		
		a_start+=1
	
	# Merging the tail of list b when list a is already merged
	while (b_end>=b_start):
		c.append(b[b_start])
		b_start+=1
	
	return c

def output_file(array,filename):
	"""
	-------------------------------------------------------------------------------
	Write the two generated lists and the merged list into a given output file name
	-------------------------------------------------------------------------------
	Preconditions:
		input_a,input_b - sorted lists generated by process 0 
		array - merged list
		filename - the output file directory
	Postconditions:
		Lists input_a,input_b and array are written to the file named filename
	-------------------------------------------------------------------------------
	"""
	file = open(filename,'w',20000)
	file.write("Merged list:")
	for process in range(len(array)):
		for partition in range(len(array[process])):
			for item in range(len(array[process][partition])):
				if (process==0 and partition==0 and item==0):
					prefix=""
				else:
					prefix=","
				file.write("{}{}".format(prefix,array[process][partition][item]))
	file.close()
	print("Sucessfully wrote results to file {}".format(filename))

def flatten_lists(list_array):
	"""
	----------------------------------------------------------------------------
	Flattening array of lists into one list
	----------------------------------------------------------------------------
	Preconditions:
		list_array - array of lists
	Postconditions:
		returns one list that includes all the given lists in the same order
	----------------------------------------------------------------------------
	"""
	list_flat = []
	for lst in list_array:
		for item in lst:
			list_flat.append(item)
	return list_flat

def test(a,b,c):
	"""
	------------------------------------------------------------------
	Testing the function by comparing lists a and b to the merged list
	------------------------------------------------------------------
	"""
	index_a = 0
	index_b = 0
	index_c = 0
	while (index_a<len(a) and index_b<len(b) and index_c<len(c)):
		if (a[index_a] == c[index_c]):
			index_a +=1
			index_c +=1
		elif (b[index_b] == c[index_c]):
			index_b +=1
			index_c +=1
		else:
			print("Value in c not found in a or b!")
			return False	
	while (index_a<len(a) and index_c<len(c)):
		if (a[index_a] == c[index_c]):
			index_a+=1
			index_c+=1
		else:
			print("Value in c not found in a, and b is tested")
			return False

	while (index_b<len(b) and index_c<len(c)):
		if (b[index_b] == c[index_c]):
			index_b+=1
			index_c+=1
		else:
			print("Value in c not found in b, and a is tested")
			return False
	if (index_a == len(a) and index_b==len(b) and index_c==len(c)):
		print("Tested and found correct")
		return True
	else:
		print("One of the lists is still with values. a={}, b={}, c={}".format(len(a),len(b),len(c)))
		return False

def main():
	# Get process size and rank
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()

	# Get size n, k=logn, r=n/k and sorted lists a and b from input
	n,k,r,a,b = get_input(rank,comm)
	#Barrier to measure start time afte input
	comm.Barrier()
	if (rank==0):
		print("Done with input stage")
		wt = MPI.Wtime()
        # Broadcasting the values of n,k and r to size-1 processes
        k = comm.bcast(k,root=0)
        n = comm.bcast(n,root=0)
        r = comm.bcast(r,root=0)
        # Broadcasting the two random lists to size-1 processes
	if (rank>0):
		a = np.empty(n,dtype=np.uint32)
		b = np.empty(n,dtype=np.uint32)
        comm.Bcast(a,root=0)
        comm.Bcast(b,root=0)
	# Dividing the partitions between the processes
	if(rank<(r%size)):
		partitions = int(math.floor(r/size))+1
	else:
		partitions = int(math.floor(r/size))		
	
	# Calculate the range dealt by the current process by multiplying the process id and it's k partition
	partitions_start = int(rank*math.floor(r/size) + min(rank,r%size))
	partitions_end = partitions_start+partitions-1
	
	#print("Process {} has {} partitions: {}-{}".format(rank,partitions,partitions_start,partitions_end))	

	a_start = []
	a_end = []
	b_start = []
	b_end = []

	# For each of the partitions, setting the values for the index in a and the last item to partition in b
	for index in range(partitions):
		a_start.insert(index,int(partitions_start*k + k*index))
		# For the last partition, the highest item in b is the last
		if (a_start[index]+k<n):
			a_end.insert(index,int(a_start[index] + k-1))
			# Finding the local upper limit in list b, by using binary search to find the minimum value bigger then the maximum local value in a
			index_b = binary_search(b,0,len(b)-1,a[a_end[index]])
			#print("Process {} index b={}".format(rank,index_b))
                        b_end.insert(index,index_b)
		else:
			a_end.insert(index,n-1)
			b_end.insert(index,n-1)	
		# The index to start iterating over b is 1 after the last index of the previous partition
                if (index>0):
                	b_start.insert(index,b_end[index-1]+1)
	
	# Sending the process last found b_end to be the neighbor's first b_start, for all but the last process
	if (rank<size-1):
		comm.send(b_end[partitions-1]+1,dest=(rank+1),tag=1)
	
	# Process 0 starts from index 0 in b
	if (rank==0):
		b_start.insert(0,0)
	# All other processes receive the first position from their left neighbor
	else:
		b_start.insert(0,comm.recv(source=(rank-1),tag=1))

	#print("Process {} a_start locations:{}".format(rank,a_start))
	#print("Process {} a_end locations:{}".format(rank,a_end))	
	#print("Process {} b_start locations:{}".format(rank,b_start))
	#print("Process {} b_end locations:{}".format(rank,b_end))

	# Array of merged lists, to be concatanated later	
	merged_lists=[]

	# Merging each partition in a loop
	for index in range(partitions):
		# Merging lists a and b in relevant positions into a local list c
        	merged_lists.append(merge(a,b,a_start[index],a_end[index],b_start[index],b_end[index]))

	# Flattening the lists into one list
	#flattened_merged = flatten_lists(merged_lists)
	
	#print("process rank {} merged_list:{}".format(rank,flattened_merged))

	# Process 0 gathers all merged lists into a new list merged, ordered by ranking index
	merged = comm.gather(merged_lists,root=0)
	
	# Process 0 now has the merged list and needs to write it to input
	if (rank==0):
		#merged_flat = []
		# Flattening the merged lists into one list
		#merged_flat = flatten_lists(merged)
		# Print the total time the program took before writing to output
		print("Total time to compute: {:2f} seconds".format(MPI.Wtime() - wt))
		# Serially making sure the merged lists match the two given lists
		#test(a,b,merged_flat)
		# Writing to output file
		output_file(merged,"/tmp/output_{}.txt".format(os.getpid()))
	
	
main()	
	

		
