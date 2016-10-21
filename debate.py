import pandas as pd
import pickle
import matplotlib.pyplot as plt
import string
import codecs


def read_and_clean():
	'''reads the debate csv and removes all Audience related text'''
	df = pd.read_csv('primary_debates_cleaned.csv')
	df.drop(['Line','URL'], axis = 1, inplace = True)
	print(df.columns)

	clinton_df = df[df['Speaker'] == 'Clinton']
	trump_df = df[df['Speaker'] == 'Trump']

	pickle.dump(clinton_df, open('clinton.pickle', 'wb'))
	pickle.dump(trump_df, open('trump.pickle', 'wb'))
	return clinton_df, trump_df

def word_counter(dataFrame):
	#return dictionaries of the candidates words and their respective counts
	word_list = list(dataFrame['Text'])

	counter = {}
	translator = str.maketrans({key: None for key in string.punctuation})
	for line in word_list:
		word_list = line.translate(translator).split()
		for word in word_list:
			word = word.lower()
			if word in counter.keys():
				counter[word] = counter[word] + 1
			else:
				counter[word] = 1
	return counter

def sentiment_analysis(word_dict, total_words):
	positive_counter = 0
	negative_counter = 0
	p_word_list = {}
	n_word_list = {}
	with open('positive-words.txt') as f:
		for line in f:
			word = str(line).rstrip()
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


clinton_df = pickle.load(open('clinton.pickle', 'rb'))
trump_df = pickle.load(open('trump.pickle', 'rb'))

'''get dictionaries of the candidate's words and their counts'''
clinton_words = word_counter(clinton_df)
trump_words = word_counter(trump_df)

clinton_total_words = sum(clinton_words.values())
trump_total_words = sum(trump_words.values())
h_pos, h_neg, h_pos_words, h_neg_words = sentiment_analysis(clinton_words, clinton_total_words)
t_pos, t_neg, t_pos_words, t_neg_words = sentiment_analysis(trump_words, trump_total_words)

print("Out of %s words and %s Hillary and Trump spoke, respectively:" % (clinton_total_words, trump_total_words))
print("Hillary's positive word rate: %0.2f" % h_pos + "%")
print("Hillary's negative word rate: %0.2f" % h_neg + "%")
print("Trump's positive word rate: %0.2f" % t_pos + "%")
print("Trump's negative word rate: %0.2f" % t_neg + "%")
print("Hillary spoke %s unique words while Trump spoke %s" % (len(clinton_words), len(trump_words)))

print(sorted(t_neg_words.items(), key = lambda x: x[1], reverse = True))
print(sorted(h_neg_words.items(), key = lambda x: x[1], reverse = True))

clinton_tuples = sorted(clinton_words.items(), key = lambda x: x[1], reverse = True)
trump_tuples = sorted(trump_words.items(), key = lambda x: x[1], reverse = True)

# print(clinton_tuples[:50])
# print(trump_tuples[:50])
# words = list(zip(*clinton_tuples))[0]
# freq = list(zip(*clinton_tuples))[1]
# x_pos = np.arange(len(words))


# plt.bar(x_pos,freq,align='center')
# plt.xticks(x_pos, words)

# plt.show()

# print(sorted(trump_words.items(), key = lambda x: x[1]))






