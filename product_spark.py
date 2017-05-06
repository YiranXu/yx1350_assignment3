from pyspark import SparkContext
from operator import mul
if __name__ == '__main__':
	sc = SparkContext("local", "product")
	result=sc.parallelize(range(1,1000)).fold(1,mul)
	print("product of all numbers from 1 to 1000: ",result)
