''' IKEA Database.
1. Download this IKEA dataset. '''

import requests
from io import StringIO
import numpy as np
import pandas as pd
from collections import defaultdict
import re

url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv"

response = requests.get(url)

if response.status_code == 200:
    data = pd.read_csv(StringIO(response.text))
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")

# Search for duplicates
# 1) Screening out duplicates by 'item_id'
unique_data = data.drop_duplicates(subset=['item_id'], keep='first').copy()

#2) Search for duplicates by 'designer'
unique_data['designer'] = unique_data['designer'].apply(lambda x: "/".join(sorted(x.split("/"))))

designer_indices = defaultdict(list)

for i, designer in enumerate(unique_data['designer']):
    sorted_designer = "/".join(sorted(designer.split("/")))
    designer_indices[sorted_designer].append(i)

duplicate_indices = [indices for indices in designer_indices.values() if len(indices) > 1]

designer_duplicates_data = unique_data.iloc[np.concatenate(duplicate_indices)].reset_index(drop=True)

# 3) Search for duplicates by 'designer' and 'depth', 'height', 'width'
same_dims_data = []

for i, row in designer_duplicates_data.iterrows():
    dims = sorted([float(x) if isinstance(x, str) and not pd.isna(x)
                   else x for x in [row['depth'], row['height'], row['width']]])
    key = tuple(dims)
    row['dims'] = str(key)
    row['link'] = re.sub(r'\d+', '', row['link'])
    same_dims_data.append(row)

same_dims_data = pd.DataFrame(same_dims_data)
same_dims_data = same_dims_data[same_dims_data['dims'].duplicated(keep=False)]

# 4) Search for duplicates by 'depth', 'height', 'width' and 'name', 'price', 'old_price', 'designer'
grouped_data = same_dims_data.groupby(['name', 'price', 'old_price', 'designer']).filter(lambda x: len(x) > 1)

#5) Find duplicates by 'name', 'price', 'old_price', 'designer', 'depth', 'height', 'width' and 'short_description'
description_duplicates = grouped_data[grouped_data['short_description'].str.contains(r'\d+')]
description_duplicates = description_duplicates.copy()
description_duplicates['short_description'] = description_duplicates['short_description'].apply(
    lambda x: "/".join(sorted(re.findall(r'\d+', x))))
description_duplicates = description_duplicates.groupby(
    ['name', 'price', 'old_price', 'designer', 'short_description']).filter(lambda x: len(x) > 1)

description_duplicates = pd.concat([description_duplicates,
                                    grouped_data[grouped_data['short_description'].str.contains(r'\d+') == False]
                                   .groupby(['name', 'price', 'old_price', 'designer'])
                                   .filter(lambda x: len(x) > 1)])

''' 6) Search for duplicates by 'name', 'price', 'old_price', 'designer', 'depth', 'height', 'width', 'short_description'
and 'link' '''
color_names = ['white', 'black', 'red', 'green', 'blue', 'yellow', 'purple', 'pink', 'brown',
               'light-grey', 'dark-grey', 'dark-brown', 'golden', 'brown', 'grey', 'beige', 'brass-colour',
               'dark-red', 'light-brown', 'light-beige', 'orange', 'light-antique', 'steel-colour',
               'anthracite', 'antique', 'turquoise', 'multicoloured-dark', 'multicolour']

color_pattern = r'\b(' + '|'.join(color_names) + r')\b'

for i, row in description_duplicates.iterrows():
    color_matches = re.findall(color_pattern, row['link'])
    if color_matches:
        new_text = '/'.join(sorted(color_matches))
        description_duplicates.loc[i, 'link'] = new_text

same_link = (description_duplicates.groupby(['name', 'price', 'old_price', 'designer', 'short_description', 'link'])
             .filter(lambda x: len(x) > 1))

# Sampling for exploratory analysis
same_link_item_ids = same_link['item_id']
analysis_data = unique_data[~unique_data['item_id'].isin(same_link_item_ids)]
print(analysis_data)