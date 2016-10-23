import math

def tf(word, line):
	return line.count(word) / len(line)

def blobs_containing(word, corpus):
	return sum(1 for line in corpus if word in line)

def idf(word, corpus):
	return math.log(len(corpus) / (1 + blobs_containing(word, corpus)))

def tf_idf(word, line, corpus):
	return tf(word, line) * idf(word, corpus)