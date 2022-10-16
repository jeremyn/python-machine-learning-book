"""
Copyright Jeremy Nation <jeremy@jeremynation.me>.
Licensed under the MIT license.

Almost entirely copied from code created by Sebastian Raschka, also licensed under the MIT license.

"""
import os
import sqlite3

DATABASE_FILENAME = os.path.join(
    os.path.dirname(__file__),
    'reviews.sqlite',
)


def check_database():
    conn = sqlite3.connect(DATABASE_FILENAME)
    c = conn.cursor()

    c.execute('SELECT * FROM review_db')
    results = c.fetchall()
    print(results)

    conn.close()


def create_database():
    conn = sqlite3.connect(DATABASE_FILENAME)
    c = conn.cursor()
    c.execute(
        'CREATE TABLE review_db (review TEXT, sentiment INTEGER, date TEXT)'
    )

    example1 = 'I love this movie'
    c.execute(
        "INSERT INTO review_db (review, sentiment, date) "
        "VALUES (?, ?, DATETIME('now'))", (example1, 1)
    )

    example2 = 'I disliked this movie'
    c.execute(
        "INSERT INTO review_db (review, sentiment, date) "
        "VALUES (?, ?, DATETIME('now'))", (example2, 0)
    )

    conn.commit()
    conn.close()


if __name__ == '__main__':
    try:
        os.remove(DATABASE_FILENAME)
    except FileNotFoundError:
        pass
    create_database()
    check_database()
