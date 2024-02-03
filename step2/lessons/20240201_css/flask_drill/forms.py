from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField


class MyForm(FlaskForm):
    name = StringField("Ім'я")
    submit = SubmitField('Відправити')
