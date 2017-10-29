"""
--------------------------------------------------------------------------------
assingment2.py
A solution in python for CP431/CP631 assignment 2
The program generates 2 n-size  arrays where n is the power of input k in base 2
It then merges the two files using parallel merge and writes the output to a file
Tested for up to k=22 -> two arrays with 4,194,304 32 bit integers each
--------------------------------------------------------------------------------
Authors: Elizabeth Gorbonos, Omer Tal, Tianran Wang
--------------------------------------------------------------------------------
"""

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
	k=i+math.floor((j-i)/2)
	if (value<=array[k]):
		# Recursion call for the lower half of the range
		return binary_search(array,i,k,value)
	else:
		# Recursion call for the higher half of the range
		return binary_search(array,k+1,j,value)

def get_input(rank,comm,n):
	"""
	-----------------------------------------------------
	Generating two lists of random n numbers
	-----------------------------------------------------
	Preconditions:
		rank - the id of the current processsor
		comm - MPI communicator instance
	Postconditinos:
		a - sorted list of random items in size n
		b - sorted list of random items in size n
	--------------------------------------------------------
	"""
	# Maximum number to generate
	LIM=3999999999
	# Generate the two lists
	a = generate_randoms(n,LIM)
	b = generate_randoms(n,LIM)
	
	return a,b
	
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
        increase=int(lim/n)
        last_value=1
	# Generate each new number as a random between the previous_value 
	# and a relative increase to ensure a sorted order
        for i in range(n):
                a[i] = random.randint(0,increase) + last_value
                last_value = a[i]
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
	----------------------------------------------------------------------------------------
	Write the two generated lists and the merged list into a given output file name
	----------------------------------------------------------------------------------------
	Preconditions: 
		array - merged list, given as an array of arrays of arrays
		(first array - each process, second array - each partition, third array - values)
		filename - the output file directory
	Postconditions:
		List in array are written to the file named filename
	----------------------------------------------------------------------------------------
	"""
	# Wrtie to fle with  a buffer of 20MB
	file = open(filename,'w',20000)
	file.write("Merged list:")
	# Iterator over all processes
	for process in range(len(array)):
		# Iterate over all partitions
		for partition in range(len(array[process])):
			# Iterate over items
			for item in range(len(array[process][partition])):
				if (process==0 and partition==0 and item==0):
					prefix=""
				else:
					prefix=","
				file.write("{}{}".format(prefix,array[process][partition][item]))
	file.close()
	print("Sucessfully wrote results to file {}".format(filename))

def test(a,b,c):
	"""
	------------------------------------------------------------------
	Testing the function by comparing lists a and b to the merged list
	------------------------------------------------------------------
	"""
	index_a = 0
	index_b = 0
	count_c = 0
	for arr_process in c:
		for arr_partition in arr_process:
			for item in arr_partition:
				if (index_a<len(a) and a[index_a] == item):
					index_a += 1
				elif (index_b<len(b) and b[index_b] == item):
					index_b +=1
				else:
					print("discrepancy found when comparing the two lists to the third. a size={}, b size={}, c size={}, c_value={}"
						.format(len(a),len(b),count_c,item))
					return False
				count_c+=1
	if (index_a == len(a) and index_b==len(b) and count_c==len(a) + len(b)):
		print("Tested and found correct")
		return True
	else:
		print("Discrepancy found when comparing the lists length. a size={}, b size={}, c size={}".format(len(a),len(b),count_c))
		return False

def main():
	# Get process size and rank
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	# Get the number k representing a power of 2 from the user
	if (len(sys.argv)<2 or not sys.argv[1].isdigit()):
		print("Please supply the program with the number of items")
		exit(1)
	k = int(sys.argv[1])
	n = 2**k
	r = int (n/k)
	# When n is not divided by k, increase the number of partitions by 1
	if (n%k>0):
		r+=1
	#Barrier to measure start time afte input
	comm.Barrier()
	if (rank==0):
		# Get the two random arrays
		a,b = get_input(rank,comm,n)
		print("Done with input stage")
		wt = MPI.Wtime()
	else:
		a = np.empty(n,dtype=np.uint32)
		b = np.empty(n,dtype=np.uint32)
	# Broadcasting the two random lists to size-1 processes
	comm.Bcast(a,root=0)
	comm.Bcast(b,root=0)
	# For one processor, no need to partition
	if (size==1):
		partitions = 1
	# Dividing the partitions between the processes
	elif(rank<(r%size)):
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

	# Array of merged lists	
	merged_lists=[]

	# Merging each partition using a loop in ascending order
	for index in range(partitions):
		# Merging lists a and b in relevant positions into a local list c
        	merged_lists.append(merge(a,b,a_start[index],a_end[index],b_start[index],b_end[index]))

	# Process 0 gathers all merged lists into a new list merged, ordered by ranking index
	merged = comm.gather(merged_lists,root=0)
	
	# Process 0 now has the merged list and needs to write it to input
	if (rank==0):
		# Print the total time the program took before writing to output
		print("Total time to compute: {:2f} seconds".format(MPI.Wtime() - wt))
		# Testing if the merged list is equal to the merged input lists serially
		test(a,b,merged)
		# Writing to output file
		output_file(merged,"/scratch/otwluq1/a2/output_{}.txt".format(os.getpid()))
	
	
main()	
	

		
