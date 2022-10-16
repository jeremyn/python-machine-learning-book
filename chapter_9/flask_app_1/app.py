"""
Copyright Jeremy Nation <jeremy@jeremynation.me>.
Licensed under the MIT license.

Almost entirely copied from code created by Sebastian Raschka, also licensed under the MIT license.

"""
from flask import (
    Flask,
    render_template,
)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('first_app.html')

if __name__ == '__main__':
    app.run(debug=True)
