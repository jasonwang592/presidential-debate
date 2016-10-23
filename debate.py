import pandas as pd
import pickle
import matplotlib.pyplot as plt
import string
import codecs
plt.style.use('fivethirtyeight')
from nltk.corpus import stopwords

def read_and_clean():
	'''reads the debate csv and removes all Audience and non-candidate related text'''
	df = pd.read_csv('primary_debates_cleaned.csv')
	df.drop(['Line','URL'], axis = 1, inplace = True)
	print(df.columns)

	clinton_df = df[df['Speaker'] == 'Clinton']
	trump_df = df[df['Speaker'] == 'Trump']

	pickle.dump(clinton_df, open('clinton.pickle', 'wb'))
	pickle.dump(trump_df, open('trump.pickle', 'wb'))
	return clinton_df, trump_df

def word_counter(dataFrame):
	'''return dictionaries of the candidates words (stripped of stopwords
	and their respective counts'''
	word_list = list(dataFrame['Text'])
	cachedStopWords = stopwords.words("english")


	counter = {}
	translator = str.maketrans({key: None for key in string.punctuation})
	for line in word_list:
		word_list = line.translate(translator).split()
		for word in word_list:
			word = word.lower()
			if word in cachedStopWords: continue
			if word in counter.keys():
				counter[word] = counter[word] + 1
			else:
				counter[word] = 1
	return counter

def sentiment_analysis(word_dict, total_words):
	'''given a dictionary of words, their freqencies and total words count,
	calculates the percentage of words that are positive or negative as defined
	by the lists from Twitter sentiment analysis. Also returns a dictionaries of
	the positive words and negative words and their respective counts'''
	positive_counter = 0
	negative_counter = 0
	p_word_list = {}
	n_word_list = {}
	with open('positive-words.txt') as f:
		for line in f:
			word = str(line).rstrip()
			if word == 'trump': continue
			if word in word_dict.keys():
				positive_counter += word_dict[word]
				p_word_list[word] = word_dict[word]
	with codecs.open('negative-words.txt','r',encoding='utf8') as g:
		for line in g:
			word = str(line).rstrip()
			if word in word_dict.keys():
				negative_counter += word_dict[word]
				n_word_list[word] = word_dict[word]

	positive_rate = (positive_counter / total_words) * 100
	negative_rate = (negative_counter / total_words) * 100

	return positive_rate, negative_rate, p_word_list, n_word_list

def sentiment_chart(p_corpus, n_corpus, l_threshold = 0, h_threshold = 100):
	'''Produces a horizontal bar chart displaying word and frequency data for a candidate
		p_corpus: dictionary of positive words
		n_corpus: dictionary of negative words
		l_threshold: the minimum number of times a word must appear for it to be plotted
		h_threshold: the maximum number of times a word can appear for it to be plotted
	'''
	p_corpus = {word:freq for word,freq in p_corpus.items() if freq > l_threshold and freq <= h_threshold}
	n_corpus = {word:freq for word,freq in n_corpus.items() if freq > l_threshold and freq <= h_threshold}
	n_corpus.update((x,y * -1) for x, y in n_corpus.items())
	p_sorted = sorted(p_corpus.items(), key = lambda x: x[1])
	n_sorted = sorted(n_corpus.items(), key = lambda x: x[1])

	corpus =  list(zip(*n_sorted))[0] + list(zip(*p_sorted))[0]
	freq = list(zip(*n_sorted))[1] + list(zip(*p_sorted))[1]

	word_index = range(0, len(corpus))
	df = pd.DataFrame()
	df['value'] = freq

	df['pos'] = [True if i > 0 else False for i in df['value']]
	plt.figure(figsize=(15,10))
	plt.barh(word_index, freq, color = df.pos.map({True: 'g', False:'r'}))
	plt.ylim([0,len(corpus)])
	plt.xlabel('Frequency')
	plt.ylabel('Word')
	plt.yticks(word_index, corpus)
	plt.tight_layout()
	plt.show()


clinton_df = pickle.load(open('clinton.pickle', 'rb'))
trump_df = pickle.load(open('trump.pickle', 'rb'))

'''get dictionaries of the candidate's words and their counts'''
clinton_words = word_counter(clinton_df)
trump_words = word_counter(trump_df)

clinton_total_words = sum(clinton_words.values())
trump_total_words = sum(trump_words.values())
h_pos, h_neg, h_pos_words, h_neg_words = sentiment_analysis(clinton_words, sum(clinton_words.values()))
t_pos, t_neg, t_pos_words, t_neg_words = sentiment_analysis(trump_words, sum(trump_words.values()))

print("Out of %s non-stopwords and %s Hillary and Trump spoke, respectively:" % (clinton_total_words, trump_total_words))
print("Hillary's positive word rate: %0.2f" % h_pos + "%")
print("Hillary's negative word rate: %0.2f" % h_neg + "%")
print("Trump's positive word rate: %0.2f" % t_pos + "%")
print("Trump's negative word rate: %0.2f" % t_neg + "%")
print("Hillary spoke %s unique words while Trump spoke %s" % (len(clinton_words), len(trump_words)))

sentiment_chart(h_pos_words, h_neg_words, 10, 50)

# print(sorted(t_pos_words.items(), key = lambda x: x[1], reverse = True))
# print(sorted(t_neg_words.items(), key = lambda x: x[1], reverse = True))
# print(sorted(h_pos_words.items(), key = lambda x: x[1], reverse = True))
# print(sorted(h_neg_words.items(), key = lambda x: x[1], reverse = True))




