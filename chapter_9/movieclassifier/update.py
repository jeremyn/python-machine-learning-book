# Copyright Jeremy Nation.
# Released under the MIT license. See included LICENSE.txt.
#
# Almost entirely copied from code created by Sebastian Raschka released under
# the MIT license. See included LICENSE.raschka.txt.
import pickle
import sqlite3

import numpy as np

from create_database import DATABASE_FILENAME
from vectorizer import (
    CLF_FILENAME,
    vect,
)

UPDATE_PICKLE_FILE = False


def update_model(db_path, model, batch_size=10000):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * FROM review_db')

    results = c.fetchmany(batch_size)
    while results:
        data = np.array(results)
        X = data[:, 0]
        y = data[:, 1].astype(int)

        classes = np.array([0, 1])
        X_train = vect.transform(X)
        model.partial_fit(X_train, y, classes=classes)
        results = c.fetchmany(batch_size)

    conn.close()


if __name__ == '__main__':
    clf = pickle.load(open(CLF_FILENAME, 'rb'))
    update_model(db_path=DATABASE_FILENAME, model=clf)
    if UPDATE_PICKLE_FILE:
        pickle.dump(clf, open(CLF_FILENAME, 'wb'), protocol=4)
