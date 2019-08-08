# Data preparation procedures

import numpy as np
import pandas as pd

from itertools import product

import tools

index_cols = ['shop_id', 'item_id', 'date_block_num']
index_cols_shop = ['shop_id', 'date_block_num']
index_cols_item = ['item_id', 'date_block_num']
index_cols_category = ['item_category_id', 'date_block_num']

data_cols = ['item_price', 'item_cnt_day']
future_index_cols = ['shop_id','item_id']

target_cols = ['target', 'revenue', 'target_shop', 'revenue_per_shop', 'target_item', 'revenue_per_item', 'target_category', 'revenue_per_category', 'target_item_price']

def all_cols():
    return index_cols + data_cols + feature_cols

def inspect_dataframe(vname, df, N=3, columns=None):
    '''
    Shows useful information from a dataframe
    '''
    
    print('\n'+vname, df.shape)
    print(df.dtypes)
    
    if columns is None:
        ddf = df
    else:
        ddf = df[columns]
    display(ddf.head(N))
    display(ddf.tail(N))

def check_nulls(df):
    null_columns=df.columns[df.isnull().any()]
    print("null_columns:",null_columns)
    for c in list(df.columns):
        s = df[c]
        print(c,len(s[s.isnull()]))
    
def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)
    
    return df

def all_fit_cols(all_data):
        # Remove index and targets (no leakages)
    fit_cols = list(all_data.columns)
    fit_cols = tools.remove_from_list(fit_cols,index_cols)
    fit_cols = tools.remove_from_list(fit_cols,target_cols)
    return fit_cols

def split_and_dates(all_data, fit_cols=None):
    
    test_month = max(all_data['date_block_num']) 
    print("Test data is where date_block_num="+str(test_month))

    val_month = max(all_data['date_block_num']) - 1
    print("Validation data is where date_block_num="+str(val_month))
    
    if fit_cols is None:
        fit_cols = all_fit_cols(all_data)
    print('\nfit_cols:',fit_cols,len(fit_cols),'\n')    

    dates = all_data['date_block_num']
    dates_train  = dates[dates <  val_month]
    dates_val = dates[dates == val_month]
    dates_test  = dates[dates == test_month]

    X_train = all_data[dates <  val_month][fit_cols]
    X_val = all_data[dates == val_month][fit_cols]
    X_test =  all_data[dates == test_month][fit_cols]

    y_train = all_data.loc[dates <  val_month, 'target'].values
    y_val = all_data.loc[dates == val_month, 'target'].values
    y_test =  all_data.loc[dates == test_month, 'target'].values

    print('train','validation','test','total')
    print(len(X_train),len(X_val),len(X_test),len(all_data))
    print(len(y_train),len(y_val),len(y_test),len(all_data))

    return X_train, y_train, dates_train, X_val, y_val, dates_val, X_test, y_test, dates_test

def split(all_data, fit_cols=None):
    X_train, y_train, dates_train, X_val, y_val, dates_val, X_test, y_test, dates_test = split_and_dates(all_data, fit_cols)
    return X_train, y_train, X_val, y_val, X_test, y_test
   
    
def initial_grid_and_aggregations(transactions,test,items,item_categories,shops):
    '''
    Returns de all_data DataFrame
    adds test data as 34th month
    '''

    if test is not None:
        
        # Prepare data with all the sales, train and test
        # the test sales are unknown, so equal to zero
        # the test data belongs to date_block_num 34, because of the competition rules

        sales = transactions[index_cols + data_cols].copy()
        sales_test = test[future_index_cols].copy()
        sales_test['date_block_num'] = 34
        sales_test['item_price'] = 0
        sales_test['item_cnt_day'] = 0
        all_sales = sales.append(sales_test,ignore_index=True)

        '''
        sales.head()
        sales.tail()
        sales_test.head()
        sales_test.tail()
        all_sales.head()
        all_sales.tail()
        '''
        print('test_sales',len(sales_test),'train_sales',len(sales),'total',len(all_sales))    
    
    else:
        all_sales = transactions
    
    # Add revenue (maybe will be useful)
    all_sales['revenue'] = all_sales['item_price']*all_sales['item_cnt_day']

    # Add item category (maybe will be useful)
    items2 = items[['item_id','item_category_id']]
    all_sales = all_sales.join(items2.set_index('item_id'), on='item_id')

    #TODO add shop_name, item_name, item_category_name
    
    # For every month we create a grid from all shops/items combinations from that month
    grid = [] 
    for block_num in all_sales['date_block_num'].unique():
        cur_shops = all_sales[all_sales['date_block_num']==block_num]['shop_id'].unique()
        cur_items = all_sales[all_sales['date_block_num']==block_num]['item_id'].unique()
        grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

    #turn the grid into pandas dataframe
    grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
    print('combinations',len(grid))

    # Create base aggregations

    #get aggregated values for (shop_id, item_id, month)
    agg_dict = {'item_cnt_day':{'target':'sum'}, 'revenue':{'revenue':'sum'}}
    gb = all_sales.groupby(index_cols,as_index=False).agg(agg_dict)
    gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values] #fix column names

    #join aggregated data to the grid
    all_data = pd.merge(grid,gb,how='left',on=index_cols).fillna(0)
    del(grid)

    #sort the data
    all_data.sort_values(['date_block_num','shop_id','item_id'],inplace=True)
    
    # add shop-month aggregates
    agg_dict = {'item_cnt_day':{'target_shop':'sum'}, 'revenue':{'revenue_per_shop':'sum'}}
    gb = all_sales.groupby(index_cols_shop,as_index=False).agg(agg_dict)
    gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
    all_data = pd.merge(all_data, gb, how='left', on=index_cols_shop).fillna(0)

    # add item-month aggregates
    agg_dict = {'item_cnt_day':{'target_item':'sum'}, 'revenue':{'revenue_per_item':'sum'}}
    gb = all_sales.groupby(index_cols_item,as_index=False).agg(agg_dict)
    gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
    all_data = pd.merge(all_data, gb, how='left', on=index_cols_item).fillna(0)

    # add itemcategory-month aggregates
    agg_dict = {'item_cnt_day':{'target_category':'sum'}, 'revenue':{'revenue_per_category':'sum'}}
    gb = all_sales.groupby(index_cols_category,as_index=False).agg(agg_dict)
    gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
    all_data = pd.merge(all_data, items2, how='left', on='item_id').fillna(0)
    all_data = pd.merge(all_data, gb, how='left', on=index_cols_category).fillna(0)

    # add item-month average price
    all_data['target_item_price'] = all_data['revenue_per_item']/all_data['target_item']
    all_data['target_item_price'] = all_data['target_item_price'].replace(-np.inf, np.nan)
    all_data['target_item_price'] = all_data['target_item_price'].replace(np.inf, np.nan)
    all_data['target_item_price'] = all_data['target_item_price'].fillna(0)

    # Downcast dtypes from 64 to 32 bit to save memory
    all_data = downcast_dtypes(all_data)
    
    return all_data

def initial_grid_and_aggregations_only_train(transactions,items,item_categories,shops):
    '''
    Returns de all_data DataFrame
    just for train data
    '''
    
    return initial_grid_and_aggregations(transactions,None,items,item_categories,shops)

def create_submission_dataframe(submission_preds, all_data, test):
    '''
    Create a submission dataframe with ID, shop_id, item_id, item_month_cnt
    '''
    
    # Append to shop_id, item_id of test data segment
    test_month = max(all_data['date_block_num']) 
    print("Test data is where date_block_num="+str(test_month))

    assert test_month == 34

    test_data = all_data[all_data['date_block_num'] == test_month][future_index_cols]
    test_data['item_cnt_month'] = submission_preds

    print(len(test_data), len(submission_preds), np.mean(all_data['target']), np.mean(submission_preds))
    #inspect_dataframe('NEXT_MONTH', test_data)
    
    #Use test data to merge ID for submission
    submission = test[['ID']+future_index_cols].merge(test_data, on=future_index_cols, how='left').fillna(0)

    print(len(test_data), len(submission), np.mean(test_data['item_cnt_month']), np.mean(submission['item_cnt_month']))

    #print(test_data[(test_data['item_id'] == 5037) & (test_data['shop_id'] == 5)])
    #print(submission[(submission['item_id'] == 5037) & (submission['shop_id'] == 5)])
    #inspect_dataframe('SUBMISSION', submission)

    return submission

