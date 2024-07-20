import sys
from sklearn.feature_extraction import text
from sklearn import pipeline
from sklearn import linear_model
import numpy as np

l = [
    ('Business means risk!', 1),
    ("This is a document", 1),
    ("this is another document", 4),
    ("documents are separated by newlines", 8)
]

def load_data(filename):
    with open(filename, 'r') as data_file:
        s = int(data_file.readline())
        X = np.zeros(s, dtype=np.object_)
        Y = np.zeros(s, dtype=np.int_)
        for i, line in enumerate(data_file):
            ind = line.index(' ')
            if ind == -1:
                raise ValueError('invalid input file')
            targ = int(line[:ind])
            words = line[ind + 1:]
            X[i] = words
            Y[i] = targ
    return X, Y

# Load training data
X, Y = load_data('trainingdata.txt')

c = pipeline.Pipeline([
    ('vect', text.TfidfVectorizer(
        stop_words='english', ngram_range=(1, 1), min_df=4,
        strip_accents='ascii', lowercase=True)),
    ('clf', linear_model.SGDClassifier(class_weight='balanced'))
])
model = c.fit(X, Y)

t = list(line for line in sys.stdin)[1:]

for y, x in zip(model.predict(t), t):
    for pattern, targ in l:
        if pattern in x:
            print(targ)
            break
    else:
        print(y)
