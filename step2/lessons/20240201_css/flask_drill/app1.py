from flask import Flask, session, render_template, make_response, url_for, redirect
from forms import MyForm
import secrets

app = Flask(__name__)

app.config['SECRET_KEY'] = secrets.token_hex(16)
print(f'SKEY: {secrets.token_hex(16)}')


@app.route('/set_session', methods=['GET', 'POST'])
def set_session():
    form = MyForm()
    if form.validate_on_submit():
        session['username'] = form.name.data
        response = make_response(redirect(url_for('get_session')))
        response.set_data(session['username'])
        return response
    return render_template('route_name.html', form=form)


@app.route('/get_session')
def get_session():
    username = session.get('username', 'Гість')
    return f'Hello, {username}'


if __name__ == '__main__':
    app.run(debug=True)
