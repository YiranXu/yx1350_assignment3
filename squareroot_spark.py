from pyspark import SparkContext
from operator import mul,add
import math
if __name__ == '__main__':
	sc = SparkContext("local", "product")
	num=sc.parallelize(range(1,1000)).map(lambda x:math.sqrt(x))
	sum=num.fold(0, add)
	print("average square root of all numbers from 1 to 1000: ",sum/1000)
