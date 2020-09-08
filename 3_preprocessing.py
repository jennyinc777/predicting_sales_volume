##################################################
# preprocessing.py
#
# the purpose for this file is to preprocess data
# from each store for modeling purposes
# 
# please run script
#
##################################################


# Imports
import pandas as pd
import numpy as np
from itertools import product
import os

# Variables
shops = [ 5,  4,  6,  3,  2,  7, 10, 12, 28, 31, 26, 25, 22, 24, 21, 15, 16, 18, 14, 19, 42, 50, 49, 53, 52, 47, 48, 57, 58, 59, 55, 56, 36, 37, 35, 38, 34, 46, 41, 44, 39, 45]
count = 0


def clean_shop(shop):

    df = pd.read_csv('./data/shops/shop_' + str(shop) + '.csv')

    df = unique_items(df)

    os.makedirs('./data/clean_shops')

    df = pivot(df)

    df.to_csv('./data/clean_shops/shop_' + str(shop) + '.csv', index = False)


def unique_items(df):

    """preprocessing a df that finds all unique items and puts all items in for every month"""

    # inspired by https://www.kaggle.com/gordotron85/future-sales-xgboost-top-3#Preprocessing

    # set variables used in 'for' loop
    indices = []
    columns = ['date_block_num', 'item_id']

    # unique item_ids and months
    id_array = list(df['item_id'].unique())
    months = list(df['date_block_num'].unique())

    # creating every item_id with every month
    indices.append(np.array(list( product(months, id_array) ), dtype=np.int16))

    # vstack
    indices = pd.DataFrame( np.vstack(indices), columns = columns )

    # setting smaller dtypes for ints in 'item_id' and 'date_block_num'
    indices['date_block_num'] = indices['date_block_num'].astype(np.int8)
    indices['item_id'] = indices['item_id'].astype(np.int16)

    # sort values and reset index
    indices.sort_values( ['date_block_num', 'item_id'], inplace = True )
    indices.reset_index(drop=True, inplace=True)

    # adding values based on indices
    all_ids = indices.merge(df[['date_block_num', 'item_id', 'item_cnt_month']], on = columns, how = 'left')
    all_ids['item_cnt_month'] = all_ids['item_cnt_month'].fillna(0).astype(np.float16)

    return all_ids


def pivot(df):
    """returns a dataframe that has columns as 'date_block_num' and index as 'item_id'"""

    # inspired by https://www.kaggle.com/dimitreoliveira/deep-learning-for-time-series-forecasting
    df = df.pivot(index='item_id', columns = 'date_block_num', values = 'item_cnt_month')

    return df


for shop in shops:
    clean_shop(shop)
    count += 1
    # progress checks
    if count % 5:
        print(f'{count}/{len(shops)}')
