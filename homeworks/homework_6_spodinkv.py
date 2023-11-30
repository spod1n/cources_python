"""
< Завдання #6 >.

Завантажте набір даних рейтингу кави та створіть візуалізації, щоб відповісти на запитання.
Ви можете використовувати додаткові типи графіків до тих, які ми дізналися на уроках, якщо ви думаєте,
що вони допоможуть відповісти на запитання.

1. Завантажте набір даних фільмів pandas
2. Перерахуйте всі стовпці набору даних та вивчіть їх типи. Вивчіть статистику з різних областей.
Опишіть, які дані ми маємо
3. Скільки всього фільмів у наборі даних?
4. Скільки фільмів міститься у наборі даних за кожний рік?
5. Покажіть детальну інформацію про найменш і найприбутковіші фільми в наборі
6. Значення "Жанр" часом здається непослідовним; спробуйте знайти ці невідповідності та виправити їх
7. Збережіть (у новий файл CSV) 10 найкращих комедій за кількістю глядачів; покажіть лише назву фільму, рік та студію
8. Використовуйте pip для встановлення двох бібліотек: lxml, MySQLconnector-python#pip
"""

import pandas as pd
import sqlite3


# %%
# CONSTANTS
URL = 'https://gist.githubusercontent.com/tiangechen/b68782efa49a16edaf07dc2cdaa855ea/' \
      'raw/0c794a9717f18b094eabab2cd6a6b9a226903577/movies.csv'

if __name__ == '__main__':
    # 1
    df_movies = pd.read_csv(URL)
    df_movies.drop_duplicates(inplace=True)

    # 2
    print(df_movies.info(), end='\n\n')
    print('First 5ve Rows:', df_movies.head(), sep='\n', end='\n\n')

    # 3
    print(f"Films Count: {len(df_movies['Film'].unique())}", end='\n\n')

    # 4
    films_by_year = df_movies.groupby('Year')['Film'].nunique().reset_index()
    print('Films Count By Year:', films_by_year.sort_values('Year', ascending=True), sep='\n', end='\n\n')

    # 5
    df_movies['WorldwideGrossFloat'] = df_movies.apply(lambda x: float(x['Worldwide Gross'].replace('$', '')), axis=1)
    print('Min Worldwide Gross:', df_movies.loc[df_movies['WorldwideGrossFloat'].idxmin()], sep='\n', end='\n\n')
    print('Max Worldwide Gross:', df_movies.loc[df_movies['WorldwideGrossFloat'].idxmax()], sep='\n', end='\n\n')

    # 6
    df_movies['Genre'] = df_movies.apply(lambda x: x['Genre'].capitalize(), axis=1)
    print(df_movies['Genre'].unique())

    # 7
    top10 = df_movies.sort_values('Audience score %', ascending=False).head(10)[['Film', 'Year', 'Lead Studio']]
    top10.to_csv('top10_movies_from_df.csv', index=False)
    print(top10)

    del df_movies, films_by_year, top10

    """ SQLite3 """
    def printer_sql_table(result_name: str, table: [tuple]) -> None:
        """ Функція для виведення даних

        :param result_name: (str) назва таблиці
        :param table: (list(tuple())) таблиця з sqlite3
        :return: None
        """
        print(result_name)

        for row in table:
            print(row)
        else:
            print('')


    DB_NAME = 'my_database.db'

    df_movies = pd.read_csv(URL)
    df_movies.drop_duplicates(inplace=True)

    with sqlite3.connect(DB_NAME) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS movies (
                Film TEXT,
                Genre TEXT,
                LeadStudio TEXT,
                AudienceScore INTEGER,
                Profitability REAL,
                RottenTomatoes INTEGER,
                WorldwideGross TEXT,
                Year INTEGER
            );''')

        cur = conn.cursor()

        df_movies.to_sql('movies', conn, if_exists='replace', index=False)

        # 2
        cur.execute('PRAGMA table_info(movies);')
        printer_sql_table('Table info:', cur.fetchall())
        cur.execute('SELECT * FROM movies LIMIT 5;')
        printer_sql_table('First 5ve Rows:', cur.fetchall())

        # 3
        cur.execute('SELECT COUNT(DISTINCT Film) AS FilmsCount FROM movies;')
        print(f'Films Count: {cur.fetchall()[0][0]}', end='\n\n')

        # 4
        cur.execute('SELECT Year, COUNT(DISTINCT Film) AS FilmsCount FROM movies GROUP BY Year ORDER BY Year DESC;')
        printer_sql_table('Films Count By Year:', cur.fetchall())

        # 5
        cur.execute('SELECT * FROM movies WHERE [Worldwide Gross] IN (SELECT MIN([Worldwide Gross]) FROM movies);')
        printer_sql_table('Min Worldwide Gross:', cur.fetchall())
        cur.execute('SELECT * FROM movies WHERE [Worldwide Gross] IN (SELECT MAX([Worldwide Gross]) FROM movies);')
        printer_sql_table('Max Worldwide Gross:', cur.fetchall())

        # 6
        cur.execute('UPDATE movies SET Genre = UPPER(SUBSTR(Genre, 1, 1)) || LOWER(SUBSTR(Genre, 2));')
        cur.execute('SELECT DISTINCT Genre FROM movies;')
        printer_sql_table('Genre:', cur.fetchall())

        # 7
        top10 = pd.read_sql_query('''
            SELECT
                Film,
                Year,
                [Lead Studio]
            FROM movies
            ORDER BY [Audience score %] DESC LIMIT 10;
            ''', conn)
        top10.to_csv('top10_movies_from_sql.csv', index=False)
        print(top10)

        conn.commit()
