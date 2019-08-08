# Features generation and encoding

import gc

import numpy as np
import pandas as pd

from data import downcast_dtypes, target_cols

cols_to_lag = ['target', 'target_shop', 'target_item', 'target_category', 'target_item_price']
shift_range = [1, 2, 3, 4, 5, 12]

def save_features(df):
    #os.remove(features/features.h5')
    #df.to_csv('features/features.csv')
    df.to_hdf('features/features.h5', key='df', mode='w', complib='zlib') #overwrite
    #np.save(arr=a, file='data/a.npy')

def load_features():
    return pd.read_hdf('features/features.h5', 'df')
    
def expanding_mean_encoding(col, ecol, group_col, df):
    '''
    Expanding Mean encoding, the df is sorted by TIME ascending
    col = column to encode
    ecol = output encoded column name
    group_col = column to group by
    df = dataframe
    '''
    
    global_mean = df[col].mean()
    
    cumsum = df.groupby(group_col)[col].cumsum() - df[col]
    cumcnt = df.groupby(group_col).cumcount()
    
    df[ecol] = cumsum/cumcnt
    df[ecol].fillna(global_mean, inplace=True)
    
    corr = np.corrcoef(df[col].values, df[ecol].values)[0][1]
    print('CORRELATION '+col+'/'+ecol+':',corr)
    
    return df

def get_lag_cols():
    lag_cols = []
    for col in cols_to_lag:
        for shift_to_apply in shift_range:
            lag_col = '{}_lag_{}'.format(col,shift_to_apply)
            lag_cols.append(lag_col)

    return lag_cols

def target_lags(index_cols, all_data):
    '''
    Create all the lag columns for targets
    '''
    lag_cols = []
    for col in cols_to_lag:
        all_data, shift_lag_cols = do_range_lags(shift_range, col, index_cols, 'date_block_num', all_data)
        lag_cols = lag_cols + shift_lag_cols
        
    return all_data, lag_cols

def do_one_lag(col, lag_col, index_cols, time_col, shift_to_apply, df, df_out = None):
    '''
    Add a lag feature column
    
    col = column to lag
    lag_column = output column name
    index_cols = index of the dataframe
    time_col = time column to lag (must be additionable to shift_to_apply)
    df = dataframe to get shift data from
    df_out = dataframe to add shifted data to (the same as df if is None)
    '''
    
    assert time_col in index_cols
    if df_out is None:
        assert lag_col not in df
    else:
        assert lag_col not in df_out
    
    dshift = df[index_cols + [col]].copy()
    dshift[time_col] = dshift[time_col] + shift_to_apply
    dshift = dshift.rename(columns={col:lag_col})
    
    if df_out is None:
        df_out = pd.merge(df, dshift, on=index_cols, how='left').fillna(0)
    else:
        df_out = pd.merge(df_out, dshift, on=index_cols, how='left').fillna(0)
    
    del(dshift)
    gc.collect()

    return df_out

def do_range_lags(shift_range, col, index_cols, time_col, df, df_out = None):
    '''
    Add lag feature columns for a shift range (list of shifts)
    
    shift_range = list of shifts to apply
    col = column to lag
    index_cols = index of the dataframe
    time_col = time column to lag (must be additionable to shift_to_apply)
    df = dataframe
    df_out = dataframe to add shift to (the same as df if is None)
    '''

    lag_cols = []
    for shift_to_apply in shift_range:
        lag_col = '{}_lag_{}'.format(col,shift_to_apply)
        lag_cols.append(lag_col)

        print(col,shift_to_apply,'->',lag_col)

        df_out = do_one_lag(col, lag_col, index_cols, time_col, shift_to_apply, df, df_out)

    return df_out, lag_cols


def shop_activity_lag(M, lag_col, index_cols_shop, df, 
                      col='target_shop', time_col='date_block_num', downcast=True):
    '''
    Lag feature with the M last months activity
    
    M = last months number to use
    lag_col = output column name
    df = dataframe to get data from
    df_out = dataframe to add lag_col to (if None, use df)
    
    '''
   
    assert lag_col not in df

    shops_info = df[index_cols_shop + [col]].copy()
    shops_info.drop_duplicates(inplace=True)
    shops_info[lag_col] = 0
    cols = list(shops_info.columns)
    cols.remove(lag_col)

    for m in range(1,M+1):
        scol = col+'_shifted'
        si_shift = shops_info[cols].copy()
        si_shift[time_col] = si_shift[time_col] + m
        si_shift = si_shift.rename(columns={col:scol})
        shops_info = pd.merge(shops_info, si_shift, on=index_cols_shop, how='left').fillna(0)
        shops_info[lag_col] = shops_info[lag_col] + shops_info[scol]
        shops_info.drop([scol], axis=1, inplace=True)
    
    #display(shops_info[shops_info['shop_id'] == 51][index_cols_shop + [col, lag_col]][-3:])
    #display(shops_info[shops_info['shop_id'] == 50][index_cols_shop + [col, lag_col]][-3:])

    df = pd.merge(df, shops_info[index_cols_shop + [lag_col]], on=index_cols_shop, how='left').fillna(0)
        
    #display(df[df['shop_id'] == 51][index_cols_shop + [lag_col]][-3:])
    #display(df[df['shop_id'] == 50][index_cols_shop + [lag_col]][-3:])

    # Downcast dtypes from 64 to 32 bit to save memory
    if downcast:
        df = downcast_dtypes(df)
    
    del(shops_info)
    gc.collect()
    
    return df
