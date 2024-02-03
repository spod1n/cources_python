from flask import Flask, render_template, redirect, url_for, request, make_response
from forms import MyForm
import secrets

app = Flask(__name__)

app.config['SECRET_KEY'] = secrets.token_hex(16)
print(f'SKEY: {secrets.token_hex(16)}')


@app.route('/set_cookie', methods=['GET', 'POST'])
def set_cookie():
    form = MyForm()

    if form.validate_on_submit():
        name_value = form.name.data
        response = make_response(redirect(url_for('get_cookie')))
        response.set_cookie('example_cookie', name_value)
        return response
    return render_template('route_name.html', form=form)


@app.route('/get_cookie')
def get_cookie():
    cookie_value = request.cookies.get('example_cookie')
    return f'Значення cookie: {cookie_value}'


if __name__ == '__main__':
    app.run(debug=True)
