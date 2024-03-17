import sqlite3

import pandas as pd


class DataHandler:
    def __init__(self, level_id: int = 1):
        self.level_id = level_id

    def __get(self):
        return DataBaseHandler(self.level_id).get_dataset_level()

    def set(self) -> list[dict]:
        return self.__get()


class DataBaseHandler:
    def __init__(self, level_id: int, dataset: pd.DataFrame = pd.DataFrame()):
        self.db_name = 'become_millionaire.db'
        self.table_name = 'Questions'
        self.dataset = dataset
        self.level_id = level_id

    def __enter__(self):
        """ Method for connecting to sqlite3 database. """
        try:
            self.connection = sqlite3.connect(self.db_name)
            return self.connection
        except sqlite3.Error as exc:
            raise ConnectionError(f'An error occurred while connecting to the database: {exc}..')
        except Exception as exc:
            raise ValueError(f'Global error: {exc}..')

    def __exit__(self, exc_type, exc_value, traceback):
        """ Method for closing the sqlite3 database at the end of the 'with' block. """
        # self.connection.commit()
        self.connection.close()

    def get_dataset_level(self) -> list[dict]:
        """ Method for getting the dataset level. """
        try:
            with self:
                dataset = pd.read_sql_query(f"SELECT * FROM {self.table_name} WHERE level = {self.level_id}",
                                            self.connection)
                answers_fields = ['Answers_A', 'Answers_B', 'Answers_C', 'Answers_D']
                dataset['Answers'] = dataset.apply(lambda row: [row[col] for col in answers_fields], axis=1)
                dataset = dataset.drop(answers_fields, axis=1)
                return dataset.to_dict('records')
        except sqlite3.Error as exc:
            raise ConnectionError(f'An error occurred while reading the database: {exc}..')
        except Exception as exc:
            raise ValueError(f'Global error: {exc}..')

    def insert_to_the_db(self):
        try:
            with self:
                return self.dataset.to_sql(self.table_name,
                                           self.connection,
                                           index=False,
                                           if_exists='replace',
                                           dtype={'Question_ID': 'INTEGER PRIMARY KEY AUTOINCREMENT'})
        except sqlite3.Error as exc:
            raise ConnectionError(f'An error occurred while reading the database: {exc}..')
        except Exception as exc:
            raise ValueError(f'Global error: {exc}..')


if __name__ == '__main__':
    df = pd.read_excel('tmp/questions.xlsx')
    DataBaseHandler(level_id=1, dataset=df).insert_to_the_db()
    # result = DataHandler(level_id=1).set()
    # print(result)
