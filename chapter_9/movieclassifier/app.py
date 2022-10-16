"""
Copyright Jeremy Nation <jeremy@jeremynation.me>.
Licensed under the MIT license.

Almost entirely copied from code created by Sebastian Raschka, also licensed under the MIT license.

"""
import pickle
import sqlite3

from flask import (
    Flask,
    render_template,
    request,
)
from wtforms import (
    Form,
    TextAreaField,
    validators,
)

from create_database import DATABASE_FILENAME
from update import update_model
from vectorizer import (
    CLF_FILENAME,
    vect,
)

app = Flask(__name__)

clf = pickle.load(open(CLF_FILENAME, 'rb'))


def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = clf.predict_proba(X).max()
    return label[y], proba


def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])


def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        "INSERT INTO review_db (review, sentiment, date) "
            "VALUES (?, ?, DATETIME('now'))",
        (document, y),
    )
    conn.commit()
    conn.close()


class ReviewForm(Form):
    moviereview = TextAreaField(
        '',
        [
            validators.DataRequired(),
            validators.length(min=15),
        ],
    )


@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)


@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    return_val = render_template('reviewform.html', form=form)
    if form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        return_val = render_template(
            'results.html',
            content=review,
            prediction=y,
            probability=round(proba*100, 2),
        )
    return return_val


@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(review, y)
    sqlite_entry(DATABASE_FILENAME, review, y)
    return render_template('thanks.html')


if __name__ == '__main__':
    update_model(db_path=DATABASE_FILENAME, model=clf)
    app.run(debug=True)
