import seaborn as sns
import pandas

import matplotlib.pyplot as plt


def get_ds_sns(name_ds: str) -> pandas.DataFrame:
    return sns.load_dataset(name_ds)


df_tips = get_ds_sns('tips')

# Task 1 - Який середній розмір чайових залежно від статі клієнта?
avg_tip_by_sex = df_tips.groupby('sex')[['total_bill', 'tip']].mean()
print(avg_tip_by_sex)

# Task 2 - Яка загальна сума рахунку залежно від дня тижня та часу доби?
total_bill_by_day_time = df_tips.groupby(['day', 'time'])['total_bill'].sum()
print(total_bill_by_day_time)

# Task 3 - Яка залежність між розміром рахунку та курінням?
sns.boxplot(x='smoker', y='total_bill', data=df_tips)
plt.xlabel('Куріння')
plt.ylabel('Розмір рахунку')
plt.title('Залежність між розміром рахунку та курінням')
plt.show()

# Task 4 - Яка залежність між розміром чайових та курінням?
sns.boxplot(x='smoker', y='tip', data=df_tips)
plt.xlabel('Куріння')
plt.ylabel('Розмір чайових')
plt.title('Залежність між розміром чайових та курінням')
plt.show()
