"""
Copyright Jeremy Nation <jeremy@jeremynation.me>.
Licensed under the MIT license.

Almost entirely copied from code created by Sebastian Raschka, also licensed under the MIT license.

"""
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

app = Flask(__name__)


class HelloForm(Form):
    sayhello = TextAreaField('', [validators.DataRequired()])


@app.route('/')
def index():
    form = HelloForm(request.form)
    return render_template('first_app.html', form=form)


@app.route('/hello', methods=['POST'])
def hello():
    form = HelloForm(request.form)
    return_val = render_template('first_app.html', form=form)
    if form.validate():
        name = request.form['sayhello']
        return_val = render_template('hello.html', name=name)
    return return_val


if __name__ == '__main__':
    app.run(debug=True)
