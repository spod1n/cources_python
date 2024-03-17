import json
import random

from flask import request, Flask, Response, session
from flask_restful import Api, Resource
from flask_session import Session

from data_handler import DataHandler

# settings
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

api = Api(app)


class BecomeMillionaireApi(Resource):
    def __init__(self):
        super().__init__()

        self.cookies_info = ('session_id', 'level_id', 'question_id')

        # get start attributes
        self.session_id = request.cookies.get('session_id', str(random.randint(100000, 999999)))
        self.level_id = int(request.cookies.get('level_id', 1))
        self.question_id = int(request.cookies.get('question_id', False))

    @staticmethod
    def create_response(*, json_response: dict, status_response: int = 200) -> Response:
        """ Method for create response in JSON format """
        return Response(response=json.dumps(json_response, ensure_ascii=False, indent=2),
                        status=status_response,
                        headers={'Content-Type': 'application/json; charset=utf-8'})

    @staticmethod
    def trim_response(response: dict) -> dict:
        """ Method to filter the dictionary for display """
        return {key: response[key] for key in ('Level', 'Value', 'Question', 'Answers')}

    def set_cookies(self, response: Response, cookies_data: tuple) -> None:
        """ Method for recording cookies """
        cookies = zip(self.cookies_info, cookies_data)

        for key, value in dict(cookies).items():
            response.set_cookie(key=key, value=str(value))

    def session_close(self, response: Response) -> None:
        """ Method for deleting cookies (if the game was finished) """
        for cookies in self.cookies_info:
            response.delete_cookie(cookies)

        session.clear()
        self.level_id, self.question_id, users_answer = (1, False, None)

    def get_question(self) -> dict:
        """ Method for getting a random question in a level. If 'question_id' is true - get this question"""
        level_questions = DataHandler(self.level_id).set()

        if self.question_id:
            for question in level_questions:
                if question['Question_ID'] == self.question_id:
                    return question
        else:
            return random.choice(level_questions)

    def response_question(self, users_answer: str = None, msg_type: int = 1) -> Response:
        """ Method create response question """
        level_question = self.get_question()
        print(f'{self.session_id=}\n{self.level_id=}\n{level_question}')

        match msg_type:
            case 1:
                response = self.create_response(json_response=self.trim_response(level_question))

                cookies_data = (self.session_id, self.level_id, level_question['Question_ID'], level_question['IsTrue'])
                self.set_cookies(response, cookies_data)
            case 2:
                answers = level_question['Answers']
                for answer in range(len(answers)):
                    if level_question['IsTrue'] in answers[answer]:
                        answers[answer] = f'✔ {answers[answer]}'
                    elif f'{users_answer.upper()}: ' in answers[answer].upper():
                        answers[answer] = f'⛝ {answers[answer]}'

                message = self.trim_response(level_question)
                message['Message'] = 'Ой, лишенько! Ви програли, бо надали неправильну відповідь...'
                response = self.create_response(json_response=message)

                self.session_close(response)
            case _:
                message = self.trim_response(level_question)
                message['Message'] = f"Я не розумію відповіді '{users_answer}'. Можливо проблема в розкладці клавіатури?"
                response = self.create_response(json_response=message)

        return response

    def get(self, users_answer: str = None):
        if isinstance(users_answer, str):
            print(f'{users_answer=}')
            if users_answer.upper() not in ['A', 'B', 'C', 'D']:
                response = self.response_question(users_answer=users_answer, msg_type=0)

            elif users_answer.upper() == self.get_question()['IsTrue']:
                self.level_id += 1
                self.question_id = False

                if self.level_id <= 15:
                    response = self.response_question(users_answer=users_answer)
                else:
                    response = self.create_response(
                        json_response={'Message': "Вітаємо! Ви відмінно знаєте Україну і виграли 1 000 000 гривень!"}
                    )

                    self.session_close(response)
            else:
                response = self.response_question(users_answer=users_answer, msg_type=2)

        else:
            response = self.response_question()

        return response


if __name__ == '__main__':
    api.add_resource(BecomeMillionaireApi, '/bm_api', '/bm_api/', '/bm_api/<string:users_answer>')
    app.run()
