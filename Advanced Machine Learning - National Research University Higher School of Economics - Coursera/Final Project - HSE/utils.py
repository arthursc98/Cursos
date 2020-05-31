import pandas as pd
import numpy as np
import scipy.sparse
from tqdm import tqdm_notebook
from itertools import product
from sklearn.feature_extraction import text
import gc
from statsmodels.tsa.stattools import adfuller

def downcast_dtypes(df, verbose = True):
    """ Try to lower memory usage"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df


def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in tqdm_notebook(lags):
        print('Lag: '+str(i)+' for '+str(col))
        shifted = tmp.copy()
        shifted['date_block_num'] += i
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        df = pd.merge(df, shifted, on = ['date_block_num','shop_id','item_id'], how = 'left')
        del shifted
        gc.collect();
    del lags
    del tmp
    return df

def tfidf_preprocess(df, col, n_features):
    tfidf = text.TfidfVectorizer(max_features=n_features)
    df[col+'_len'] = df[col].map(len)
    df[col+'_word_count'] = df[col].map(lambda x: len(str(x).split(' ')))
    txtFeatures = pd.DataFrame(tfidf.fit_transform(df[col]).toarray())
    cols = txtFeatures.columns
    for i in range(n_features):
        df[col+'_tfidf_' + str(i)] = txtFeatures[cols[i]]
    
    df = downcast_dtypes(df)

    gc.collect();
    
    del txtFeatures
    del tfidf
    del cols
    
    return df

def encoded(matrix, cols, name, lags):
    a = matrix.copy()
    del matrix
    group = a.groupby(cols).agg({'item_cnt_month': ['mean']})
    group.columns = [name]
    group.reset_index(inplace=True)

    gc.collect();
    a = a.join(group.set_index(cols), on = cols)
    del group
    gc.collect();
    a[name] = a[name].astype(np.float16)
    a = lag_feature(a, lags, name)
    del lags
    gc.collect();
    a.drop([name], axis=1, inplace=True)
    gc.collect();
    
    return a


def create_lagged_features(matrix):
    """
    1) monthly
    2) monthly every item
    3) monthly every shop
    4) monthly every item category
    5) monthly every shop every item category
    6) monthly every shop every type category
    7) monthly every type category
    8) monthly every city category
    9) monthly every shop every subtype category
    10) monthly every item every city category
    11) monthly every subtype category
    """
    print('1.')
    matrix1 = encoded(matrix, ['date_block_num'], 'date_avg_item_cnt', [1])
    matrix1 = downcast_dtypes(matrix1)
    gc.collect();
    print('2.')
    matrix1 = encoded(matrix1, ['date_block_num', 'item_id'], 'date_item_avg_item_cnt', [1,2,3,6,12])
    matrix1 = downcast_dtypes(matrix1)
    gc.collect();
    print('3.')
    matrix1 = encoded(matrix1, ['date_block_num', 'shop_id'], 'date_shop_avg_item_cnt', [1,2,3,6,12])
    matrix1 = downcast_dtypes(matrix1)
    gc.collect();
    print('4.')
    matrix1 = encoded(matrix1, ['date_block_num', 'item_category_id'], 'date_category_avg_item_cnt', [1])
    matrix1 = downcast_dtypes(matrix1)
    gc.collect();
    print('5.')
    matrix1 = encoded(matrix1, ['date_block_num', 'shop_id', 'item_category_id'], 'date_shop_category_avg_item_cnt', [1])
    matrix1 = downcast_dtypes(matrix1)
    gc.collect();
    print('6.')
    matrix1 = encoded(matrix1, ['date_block_num', 'shop_id', 'type_code'], 'date_shop_type_avg_item_cnt', [1])
    matrix1 = downcast_dtypes(matrix1)
    gc.collect();
    print('7.')
    matrix1 = encoded(matrix1, ['date_block_num', 'type_code'], 'date_type_avg_item_cnt', [1])
    matrix1 = downcast_dtypes(matrix1)
    gc.collect();
    print('8.')
    matrix1 = encoded(matrix1, ['date_block_num', 'city_code'], 'date_city_avg_item_cnt', [1])
    matrix1 = downcast_dtypes(matrix1)
    gc.collect();
    print('9.')
    matrix1 = encoded(matrix1, ['date_block_num', 'shop_id', 'subtype_code'], 'date_shop_subtype_avg_item_cnt', [1])
    matrix1 = downcast_dtypes(matrix1)
    gc.collect();
    print('10.')
    matrix1 = encoded(matrix1, ['date_block_num', 'item_id', 'city_code'], 'date_item_city_avg_item_cnt', [1])
    matrix1 = downcast_dtypes(matrix1)
    gc.collect();
    print('11.')
    matrix1 = encoded(matrix1, ['date_block_num', 'subtype_code'], 'date_subtype_avg_item_cnt', [1])
    matrix1 = downcast_dtypes(matrix1)
    gc.collect();
    
    return matrix1


def print_df_info(df):
    print(f"---------------------------------")
    print(f"        Rows: {df.shape[0]}")
    print(f"        Columns: {df.shape[1]}")
    print(f"        NaN Values: {df.isna().sum().sum()}")
    print(f"        Missing Values: {df.isnull().sum().sum()}")
    print(f"        Duplicated Rows: {df.duplicated().sum()}")
    print(f"---------------------------------")

def adfuller_test(series, col):
    # Lag between every datapoint
    stationary = series[col]
    stationary.index = series.index[:]

    # Hypothesis_testing
    hypothesis_testing = adfuller(stationary)
    test_stat = hypothesis_testing[0]
    print('Test Statistic: {}'.format(test_stat))
    print('Critical Value for 1%: {}'.format(hypothesis_testing[4]['1%']))
    print('Critical Value for 5%: {}'.format(hypothesis_testing[4]['5%']))

def create_df_from_eda():
    extension = '.csv'
    join_file_extension = lambda x, y: x+y
    sales_train = pd.read_csv(join_file_extension('sales_train',extension))
    items = pd.read_csv(join_file_extension('items',extension))
    item_categories = pd.read_csv(join_file_extension('item_categories',extension))
    shops = pd.read_csv(join_file_extension('shops',extension))
    df = sales_train.merge(items, on = 'item_id', how = 'left').merge(shops, on = 'shop_id', how = 'left').merge(item_categories, on = 'item_category_id', how = 'left')
    cols = ['date','date_block_num','shop_id','item_id','item_cnt_day']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by = 'date')
    df = df[df.item_price<100000]
    df = df[df.item_cnt_day<1001]
    df = df.drop_duplicates(subset = cols)
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['sellout'] = df['item_price'] * df['item_cnt_day']
    del sales_train
    del items
    del item_categories
    del shops
    del cols

    return df