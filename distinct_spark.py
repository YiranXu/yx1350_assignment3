#to execute locally:PATH=$PATH://Users/yiranxu/Documents/spark-2.1.1-bin-hadoop2.7/bin
from pyspark import SparkContext
from operator import add
import re

# remove any non-words and split lines into separate words
# finally, convert all words to lowercase
def splitter(line):
    line = re.sub(r'^\W+|\W+$', '', line)
    return map(str.lower, re.split(r'\W+', line))

if __name__ == '__main__':
	sc = SparkContext("local", "wordcount")
	text = sc.textFile('pg2701.txt')
	words = text.flatMap(splitter)
	words_mapped = words.map(lambda x: (x,1))
	sorted_map = words_mapped.sortByKey()
	sort_map2=sorted_map.groupByKey()
	print("number of distinct words",sort_map2.count())
