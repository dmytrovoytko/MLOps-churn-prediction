# import warnings # supress warnings
# warnings.filterwarnings('ignore')

import os
import pickle

import numpy as np
import pandas as pd
from settings import DATA_DIR, DATASET_NUM, MODEL_DIR, REMOVE_OUTLIERS, TARGET
from sklearn.preprocessing import OrdinalEncoder

# Display all columns
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


from settings import DEBUG  # isort:skip

DEBUG = True  # True # False # override global settings

from utils import S3_ENDPOINT_URL, S3_BUCKET  # isort:skip


def model_dir(dataset_num=DATASET_NUM):
    return f'./model/{dataset_num}/'  # ! with '/' at the end!


def load_dataset(dataset_num=DATASET_NUM):
    print(f'\nLoading dataset [{dataset_num}]...')
    if dataset_num == 1:
        file_name = 'E Commerce Dataset.xlsx'
        data = pd.read_excel(DATA_DIR + file_name, sheet_name='E Comm')
        if DEBUG:
            data_dict = pd.read_excel(DATA_DIR + 'E Commerce Dataset.xlsx', sheet_name='Data Dict')
            data.to_csv(DATA_DIR + file_name + '.csv', encoding='utf-8', index=False)
            data_dict.to_csv(DATA_DIR + file_name[:-5] + '_dict.csv', encoding='utf-8', index=False)
    elif dataset_num == 2:
        file_name = 'dqlab_telco_final.csv'
        data = pd.read_csv(DATA_DIR + file_name)
    else:
        print('Load data Error: incorrect dataset number:', dataset_num)
        return pd.DataFrame()

    if DEBUG:
        print(f' Loaded {data.shape[0]} records.')
        data.info()
        print(data.head())

    return data


def determine_outlier_thresholds_iqr(df, col_name, th1=0.25, th3=0.75):
    # for removing outliers using Interquartile Range or IQR
    quartile1 = df[col_name].quantile(th1)
    quartile3 = df[col_name].quantile(th3)
    iqr = quartile3 - quartile1
    upper_limit = quartile3 + 1.5 * iqr
    lower_limit = quartile1 - 1.5 * iqr
    return lower_limit, upper_limit


def determine_outlier_thresholds_sdm(df, col_name, scale):
    # for removing outliers using the Standard deviation method
    df_mean = df[col_name].mean()
    df_std = df[col_name].std()
    upper_limit = df_mean + scale * df_std
    lower_limit = df_mean - scale * df_std
    return lower_limit, upper_limit


def print_missing_values_table(data, na_name=False):
    na_columns = [col for col in data.columns if data[col].isnull().sum() > 0]
    n_miss = data[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (data[na_columns].isnull().sum() / data.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def preprocess_df(data):
    print('\nPreprocessing data...')
    target = TARGET
    total_rows_number = data.shape[0]

    columns = data.columns.to_list()

    # 1. drop useless columns
    useless_columns = [
        'CustomerID',
        'customerID',
        'UpdatedAt',
    ]
    for col in useless_columns:
        if col in columns:  # exists in df
            if DEBUG and col.lower() == 'customerid':
                print(f'Total rows: {total_rows_number}, unique {col}s: {data[col].nunique()}')
            data.drop([col], axis=1, inplace=True)

    # 2. inspect categorical columns
    categorical = data.dtypes[data.dtypes == 'object'].keys()
    if target in categorical:
        # specifics of DATASET_NUM==2, TARGET must be encoded to 0/1 BEFORE using OrdinalEncoder
        data.loc[data[target] == 'No', target] = 0
        data.loc[data[target] == 'Yes', target] = 1
        data[target] = data[target].astype(int)
        # update categorical
        categorical = data.dtypes[data.dtypes == 'object'].keys()
        if DEBUG:
            print(target, 'encoded')

    if DEBUG and target in columns:
        print(f'\nCategorical columns: {list(categorical)}')
        # print distribution for each
        for col in categorical:
            print('\n by', data[col].value_counts().to_string())

        if len(data) > 10:
            corr = data.corr(numeric_only=True)[target]
            print(f'\nCorrelation to {target}:\n{corr.to_string()}')

    # 3. inspect missing values
    nan_cols = data.columns[data.isnull().any()].to_list()
    if DEBUG and nan_cols:
        # list of columns with missing values and its percentage
        print(f'\nColumns with nulls:\n{nan_cols}')
        print_missing_values_table(data, na_name=True)

    # 4. fix missing values - fill with median values
    # fix_cols = ["Tenure","DaySinceLastOrder","OrderAmountHikeFromlastYear","OrderCount",
    #             "CouponUsed","HourSpendOnApp","WarehouseToHome",]
    if nan_cols:
        if DEBUG:
            print(f'\nFixing missing values...')
        data.loc[:, nan_cols] = data.loc[:, nan_cols].fillna(data.loc[:, nan_cols].median())

    if DEBUG:
        nan_cols = data.columns[data.isnull().any()].to_list()
        print(f' Any columns with nulls left? {nan_cols}')

    if REMOVE_OUTLIERS:
        outlier_cols = ['Tenure', 'tenure']
        for col in outlier_cols:
            if col in columns:
                # print(f"\nRemoving {col} outliers using the Standard deviation method")
                # lower, upper = determine_outlier_thresholds_sdm(data, col, 6) # 4
                # print(" upper limit:", upper)
                # print(" lower limit:", lower)
                print(f"\nRemoving {col} outliers using IQR")
                lower, upper = determine_outlier_thresholds_iqr(data, col, th1=0.05, th3=0.95)
                print(" upper limit:", upper)
                print(" lower limit:", lower)
                data = data[(data[col] >= lower) & (data[col] <= upper)]

    print(
        f'\nFinal number of records: {data.shape[0]} / {total_rows_number} =',
        f'{data.shape[0]/total_rows_number*100:05.2f}%\n',
    )
    return data


def enc_save(enc, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(enc, f)


def enc_load(file_name):
    # if S3_ENDPOINT_URL:
    #     try:
    #         s3_download_file(S3, S3_BUCKET, 'model/encoder.pkl', f'{MODEL_DIR}encoder.pkl', verbose=DEBUG)
    #     except Exception as e:
    #         print(f'! Error S3_ENDPOINT_URL {S3_ENDPOINT_URL}, S3_BUCKET {S3_BUCKET}\n', e)

    with open(file_name, 'rb') as f:
        enc = pickle.load(f)
        return enc
    return OrdinalEncoder()


def preprocess_data(df, ord_enc, dataset_num=DATASET_NUM, fit_enc=False):
    # fix missing values, remove outliers
    df = preprocess_df(df)

    # encode categorical
    categorical_features = df.select_dtypes(exclude=[np.number]).columns
    if len(categorical_features):
        if DEBUG:
            print('OrdinalEncoder categorical_features:', list(categorical_features))
        # import ordinal encoder from sklearn
        # ord_enc = OrdinalEncoder()
        if fit_enc:
            # Fit and Transform the data
            df[categorical_features] = ord_enc.fit_transform(df[categorical_features])
            # ensure directoty exists
            os.makedirs(model_dir(dataset_num), exist_ok=True)
            enc_save(ord_enc, f'{model_dir(dataset_num)}encoder.pkl')
            if DEBUG:
                print(' OrdinalEncoder categories:', ord_enc.categories_)
        else:
            # Only Transform the data (using pretrained encoder)
            df[categorical_features] = ord_enc.transform(df[categorical_features])

    columns = df.columns.to_list()
    if DEBUG and TARGET in columns and len(df) > 10:
        corr = df.corr(numeric_only=True)[TARGET]
        print(f'\nCorrelation-2 to {TARGET}:\n{corr.to_string()}')

    return df


def load_data(dataset_num=DATASET_NUM):
    if dataset_num == 1:
        df = load_dataset(1)
        ord_enc = OrdinalEncoder()
        df = preprocess_data(df, ord_enc, 1, fit_enc=True)
    elif dataset_num == 2:
        df = load_dataset(2)
        ord_enc = OrdinalEncoder()
        df = preprocess_data(df, ord_enc, 2, fit_enc=True)
    else:
        print('Load data Error: incorrect dataset number:', dataset_num)
        return pd.DataFrame()

    return df


if __name__ == '__main__':
    # quick tests
    for dataset_num in [1, 2]:
        # full dataset
        df = load_data(dataset_num)  # load dataset and train encoder
        df.info()
        print(f'\nfull dataset {dataset_num} head(5)\n', df.head(5).to_string())

        # mini dataset 1
        df = load_dataset(dataset_num).head(10)
        # df1 = preprocess_df(df)
        print(f'\nmini dataset {dataset_num}\n', df.head(1).to_string())
        # loading trained encoder
        ord_enc = enc_load(f'{model_dir(dataset_num)}encoder.pkl')
        df = preprocess_data(df, ord_enc, 1, fit_enc=False)
        print(f'\nmini encoded {dataset_num}\n', df.head(1).to_string())
