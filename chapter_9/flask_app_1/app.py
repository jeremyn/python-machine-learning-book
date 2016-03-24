# Copyright Jeremy Nation.
# Released under the MIT license. See included LICENSE.txt.
#
# Almost entirely copied from code created by Sebastian Raschka released under
# the MIT license. See included LICENSE.raschka.txt.
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
