import os
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (
        re.sub('[\W]+', ' ', text.lower()) +
        ' '.join(emoticons).replace('-', '')
    )
    return text


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [PorterStemmer().stem(word) for word in text.split()]


def run_grid_search():
    df = pd.read_csv(os.path.join('datasets', 'movie_data.csv'))
    X_train = df.loc[:25000, 'review'].values
    y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    y_test = df.loc[25000:, 'sentiment'].values

    stop = stopwords.words('english')

    tfidf = TfidfVectorizer(
        strip_accents=None,
        lowercase=None,
        preprocessor=None,
    )

    param_grid = [
        {
            'vect__ngram_range': [(1, 1)],
            'vect__stop_words': [stop, None],
            'vect__tokenizer': [tokenizer, tokenizer_porter],
            'clf__penalty': ['l1', 'l2'],
            'clf__C': np.logspace(0, 2, num=3),
        },
        {
            'vect__ngram_range': [(1, 1)],
            'vect__stop_words': [stop, None],
            'vect__tokenizer': [tokenizer, tokenizer_porter],
            'vect__use_idf': [False],
            'vect__norm': [None],
            'clf__penalty': ['l1', 'l2'],
            'clf__C': np.logspace(0, 2, num=3),
        },
    ]

    lr_tfidf = Pipeline([
        ('vect', tfidf),
        ('clf', LogisticRegression(random_state=0)),
    ])

    gs_lr_tfidf = GridSearchCV(
        lr_tfidf,
        param_grid,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1,
    )

    gs_lr_tfidf.fit(X_train, y_train)

    print(gs_lr_tfidf)
    print("Best parameter set: %s" % gs_lr_tfidf.best_params_)
    print("CV accuracy: %.3f" % gs_lr_tfidf.best_score_)

    clf = gs_lr_tfidf.best_estimator_
    print("Test accuracy: %.3f" % clf.score(X_test, y_test))


def work_with_simple_bag_of_words():
    count = CountVectorizer()
    docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining and the weather is sweet',
    ])
    bag = count.fit_transform(docs)
    print(count.vocabulary_)
    print(bag.toarray())

    np.set_printoptions(precision=2)
    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    print(tfidf.fit_transform(bag).toarray())

    tf_is = 2
    n_docs = 3
    idf_is = np.log((n_docs+1) / (3+1))
    tfidf_is = tf_is * (idf_is + 1)
    print("tf-idf of term 'is' = %.2f" % tfidf_is)

    tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
    raw_tfidf = tfidf.fit_transform(bag).toarray()[-1]
    print(raw_tfidf)

    l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
    print(l2_tfidf)


if __name__ == '__main__':
    np.random.seed(0)
    # work_with_simple_bag_of_words()
    run_grid_search()
