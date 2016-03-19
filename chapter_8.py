import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
)


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
    work_with_simple_bag_of_words()
