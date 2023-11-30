"""
< Завдання #7 >.

1. Прочитайте набір даних паверліфтингу в pandas DataFrame

2. Знайдіть рекорди кожної статі та кожного підрозділу в кожній із 3 вправ
(жим лежачи (best3bench_kg), присідання (best3squat_kg) та станова тяга (best3deadlift_kg)).

3. Порахуйте кількість перемог кожного учасника, приймаючи за перемогу 1-е місце.
Збережіть результати у вторинному DataFrame.

4. Використайте DataFrame з вправи #3, щоб показати для кожної комбінації підрозділи та статі,
учасника з найбільшою кількістю перемог серед учасників, які будь-коли брали участь у цьому підрозділі.
"""

import sqlite3
import pandas as pd


URL = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-10-08/ipf_lifts.csv'
DB_NAME = 'my_database.db'

if __name__ == '__main__':
    # 1
    df_powerlifting = pd.read_csv(URL)

    with sqlite3.connect(DB_NAME) as conn:
        df_powerlifting.to_sql('ipf_lifts', conn, if_exists='replace')

        # 2
        df_records = pd.read_sql('''
            SELECT
                sex,
                division,
                MAX(best3bench_kg) AS bench,
                MAX(best3squat_kg) AS squat,
                MAX(best3deadlift_kg) AS deadlift
            FROM ipf_lifts
            GROUP BY
                sex,
                division;
            ''', conn)

        print('Records By Division:', df_records, sep='\n', end='\n\n')
        print('Records By Sex:', df_records.groupby('sex')[['bench', 'squat', 'deadlift']].max(), sep='\n', end='\n\n')

        # 3
        df_winners = pd.read_sql("""
            SELECT
                name,
                COUNT(place) AS wins_cnt
            FROM ipf_lifts
            WHERE place = '1'
            GROUP BY name;
            """, conn)
        print('The Number Of Victories Of Each Participant:', df_winners, sep='\n', end='\n\n')

        # 4
        wins_cnt = pd.read_sql("""
            SELECT
                division,
                weight_class_kg,
                sex,
                name,
                COUNT(place) AS wins_num
            FROM ipf_lifts
            WHERE place = '1'
            GROUP BY
                division,
                weight_class_kg,
                sex,
                name
            """, conn)

        div_best = wins_cnt.merge(df_winners, on='name')
        div_best.sort_values('wins_num', ascending=False, inplace=True)
        div_best.drop_duplicates(['division', 'weight_class_kg', 'sex'], keep='first', inplace=True)
        div_best = div_best[['division', 'weight_class_kg', 'sex', 'name', 'wins_num', 'wins_cnt']].reset_index(drop=True)
        div_best = div_best[div_best.division.notna()]
        print(div_best)

        conn.commit()
