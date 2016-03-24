# Copyright Jeremy Nation.
# Released under the MIT license. See included LICENSE.txt.
#
# Almost entirely copied from code created by Sebastian Raschka released under
# the MIT license. See included LICENSE.raschka.txt.
import os
import pickle
import re

from sklearn.feature_extraction.text import HashingVectorizer


_PKL_OBJECTS_DIR = os.path.join(
    os.path.dirname(__file__),
    'pkl_objects',
)
if not os.path.exists(_PKL_OBJECTS_DIR):
    os.makedirs(_PKL_OBJECTS_DIR)

CLF_FILENAME = os.path.join(_PKL_OBJECTS_DIR, 'classifier.pkl')

_STOPWORDS_FILENAME = os.path.join(_PKL_OBJECTS_DIR, 'stopwords.pkl')

try:
    _stop = pickle.load(open(_STOPWORDS_FILENAME, 'rb'))
except FileNotFoundError:
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    pickle.dump(stop, open(_STOPWORDS_FILENAME, 'wb'), protocol=4)


def _tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (
        re.sub('[\W]+', ' ', text.lower()) +
        ' '.join(emoticons).replace('-', '')
    )
    tokenized = [w for w in text.split() if w not in _stop]
    return tokenized


vect = HashingVectorizer(
    decode_error='ignore',
    n_features=2**21,
    preprocessor=None,
    tokenizer=_tokenizer,
)


def get_minibatch(doc_stream, size):
    docs = []
    y = []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        docs = None
        y = None
    return docs, y


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            text = line[:-3]
            label = int(line[-2])
            yield text, label


def train_and_pickle_classifier():
    import numpy as np
    from sklearn.linear_model import SGDClassifier

    clf = SGDClassifier(loss='log', random_state=1, n_iter=1)

    csv_filename = os.path.join('datasets', 'movie_data.csv')
    doc_stream = stream_docs(path=csv_filename)

    classes = np.array([0, 1])
    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if X_train is None:
            break
        else:
            X_train = vect.transform(X_train)
            clf.partial_fit(X_train, y_train, classes=classes)

    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print("Test accuracy: %.3f" % clf.score(X_test, y_test))

    clf = clf.partial_fit(X_test, y_test)

    pickle.dump(clf, open(CLF_FILENAME, 'wb'), protocol=4)


if __name__ == '__main__':
    import numpy as np
    clf = pickle.load(open(CLF_FILENAME, 'rb'))

    label = {0: 'negative', 1: 'positive'}
    example = ['I love this movie']
    X = vect.transform(example)
    print("Prediction: %s" % label[clf.predict(X)[0]])
    print("Probability: %.2f%%" % np.max(clf.predict_proba(X) * 100))
