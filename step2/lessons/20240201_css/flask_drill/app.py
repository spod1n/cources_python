from flask import Flask, render_template, redirect, url_for, request, make_response
from forms import MyForm
import secrets

app = Flask(__name__)

app.config['SECRET_KEY'] = secrets.token_hex(16)
print(f'SKEY: {secrets.token_hex(16)}')


@app.route('/route_name', methods=['GET', 'POST'])
def route_name():
    form = MyForm()

    if form.validate_on_submit():
        name = form.name.data
        return redirect(url_for('success'))
    return render_template('route_name.html', form=form)


@app.route('/success')
def success():
    return 'Форма відправлена'


@app.route('/set_cookie/<value>')
def set_cookie(value):
    response = make_response('Cookie встановлено')
    response.set_cookie('example_cookie', value)
    return response


@app.route('/get_cookie')
def get_cookie():
    cookie_value = request.cookies.get('example_cookie')
    return f'Значення cookie: {cookie_value}'


if __name__ == '__main__':
    app.run(debug=True)
