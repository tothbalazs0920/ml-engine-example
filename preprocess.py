from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
from sklearn.datasets import fetch_20newsgroups


def preprocess(text, target, x_file_name, y_file_name, tokenizer):
    y = np.zeros([len(target), 4])

    for i, target in enumerate(target):
        if target == 0:
            y[i] = [1, 0, 0, 0]
        elif target == 1:
            y[i] = [0, 1, 0, 0]
        elif target == 2:
            y[i] = [0, 0, 1, 0]
        else:
            y[i] = [0, 0, 0, 1]

    encoded_docs = tokenizer.texts_to_matrix(text, mode='count')
    np.savetxt(x_file_name, encoded_docs, delimiter="\t")
    np.savetxt(y_file_name, y, delimiter="\t")


categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
remove = ('headers', 'footers', 'quotes')

news_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

news_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)

text_train, target_train = news_train.data, news_train.target

tokenizer = Tokenizer(num_words=300)
tokenizer.fit_on_texts(news_train)

with open('data/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

preprocess(text_train, target_train, "data/x_train.csv",
           "data/y_train.csv", tokenizer)

text_test, target_test = news_test.data, news_test.target
preprocess(text_test, target_test, "data/x_test.csv",
           "data/y_test.csv", tokenizer)
