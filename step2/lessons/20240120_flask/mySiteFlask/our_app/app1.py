from flask import Flask
from flask import render_template, jsonify, request
# from flask_sqlalchemy import SQLAlchemy
from flask_pymongo import MongoClient

app = Flask(__name__)
app.config['Mongo_URI'] = MongoClient('mongodb://localhost:27017/books2_db')



# db = SQLAlchemy(app)


# def plus1(num):
#     num += 1
#     return num
#
#
# @app.route('/hello/<name>')
# def hello(name):
#     return render_template('hello.html', name=name)
#
#
# @app.route(f'/index/<number>')
# def index(number):
#     return render_template('index.html', number=plus1(int(number)))
#
#
# @app.route('/get_updated_number')
# def get_updated_number():
#     updated_number = plus1(int(request.args.get('number', 0)))

# return jsonify({'updatedNumber': updated_number})

# num = 0
#
#
# @app.route('/index')
# def home():
#     global num
#     num += 1
#     return render_template('index1.html', index=num)

# @app.route('get_data')
# def get_data():
#     data_from_mongo = mongo.db.collection_name.find_one({'key': 'value'})
#     return jsonify({'result': data_from_mongo})


if __name__ == '__main__':
    app.run(debug=True)
