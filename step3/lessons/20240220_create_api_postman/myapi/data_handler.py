import json

dummy_data = {
    "id": 0,
    "title": "",
    "text": "",
    "link": ""
}


class DataHandler:
    @staticmethod
    def __get():
        return FileHandler().read()

    def set(self) -> list[dict]:
        return eval(self.__get())


class FileHandler:
    filename = 'dataset.json'

    def __init__(self):
        self.mode_read = open(file=self.filename, mode='r', encoding='UTF-8')

    def read(self) -> str:
        return self.mode_read.read()

    def __del__(self):
        self.mode_read.close()