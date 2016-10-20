import pandas as pd
import pickle
import matplotlib.pyplot as plt
import string
import numpy as np


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


clinton_df = pickle.load(open('clinton.pickle', 'rb'))
trump_df = pickle.load(open('trump.pickle', 'rb'))

#return dictionaries of the candidates words and their respective counts
#calling len(dict) shows us the size of one candidate's vocabulary to another's
clinton_words = word_counter(clinton_df)
trump_words = word_counter(trump_df)

clinton_total_words = sum(clinton_words.values())
trump_total_words = sum(trump_words.values())
print(clinton_total_words, trump_total_words)

clinton_tuples = sorted(clinton_words.items(), key = lambda x: x[1], reverse = True)
trump_tuples = sorted(trump_words.items(), key = lambda x: x[1], reverse = True)

print(clinton_tuples[:50])
print(trump_tuples[:50])
print(len(clinton_words), len(trump_words))
# words = list(zip(*clinton_tuples))[0]
# freq = list(zip(*clinton_tuples))[1]
# x_pos = np.arange(len(words))


# plt.bar(x_pos,freq,align='center')
# plt.xticks(x_pos, words)

# plt.show()

# print(sorted(trump_words.items(), key = lambda x: x[1]))






