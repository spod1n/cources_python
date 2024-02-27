import json

from flask import Flask
from flask_restful import Api, Resource, reqparse

from data_handler import DataHandler

app = Flask(__name__)
api = Api(app)


class NewsApi(Resource):
    def get(self, id: int = -2):
        data = DataHandler().set()
        for res in data:
            if res['id'] == id or res['id'] == -1:
                code = 200 if res['id'] == id else 400
                if code == 200:
                    break
        return res, code

    def post(self, id: int = -1):
        args = ['title', 'text']
        for x in args:
            parser = reqparse.RequestParser()
            params = parser.parse_args()

        data = DataHandler().set()
        for x in data:
            if x['id'] == id:
                x = {
                    'id': int(id),
                    'title': params['title'],
                    'text': params['text']
                }
            else:
                code = 404
                return 201, code

    def put(self, id: int = -2):
        args = ['title', 'text']
        for x in args:
            parser = reqparse.RequestParser()
            params = parser.parse_args()

        data = DataHandler().set()
        for x in data:
            if x['id'] == id or x['id'] == -2:
                if x['id'] == id:
                    x = {
                        'id': int(id),
                        'title': params['title'],
                        'text': params['text']
                    }
                    code = 201
                return x, code

    def delete(self): ...


if __name__ == '__main__':
    api.add_resource(NewsApi, '/news', '/news/', '/news/<int:id>')
    app.run()
