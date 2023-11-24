import pandas as pd

df = pd.read_csv("ikea.csv")
print(f"Довжина оригінальна {df.__len__()}")
df = df.rename(columns={'Unnamed: 0': 'index'})
unique = df.drop_duplicates()
print(f"Довжина без дуплікатів по всьому {unique.__len__()}")

# cleaning duplicates in item_id
unique_df = unique.drop_duplicates(subset=['item_id'], keep='first').copy()
print(f"Довжина без дуплікатів по айді {unique_df.__len__()}")

# шукаю викиди по ціні і зразу видаляю
def remove_outliers(df):
    # Calculate quartiles and interquartile range (IQR)
    q1 = df['price'].quantile(0.25)
    q3 = df['price'].quantile(0.75)
    iqr = q3 - q1

    # Calculate lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter rows without outliers
    return df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# Remove outliers based on 'price' across the entire DataFrame
unique_df_no_outliers = remove_outliers(unique_df)

# Print the length after removing outliers
print(f"Довжина без викидів по ціні: {unique_df_no_outliers.__len__()}")
