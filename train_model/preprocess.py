# import warnings # supress warnings
# warnings.filterwarnings('ignore')

import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from settings import DATA_DIR, DATASET_NUM, MODEL_DIR, REMOVE_OUTLIERS, TARGET

from settings import DEBUG # isort:skip
# DEBUG = True # True # False # override global settings

def load_dataset(dataset_num=DATASET_NUM):
    print('\nLoading dataset...')
    if dataset_num==1:
        file_name = 'E Commerce Dataset.xlsx'
        data = pd.read_excel(DATA_DIR + file_name, sheet_name='E Comm')
        if DEBUG:
            data_dict = pd.read_excel(DATA_DIR + 'E Commerce Dataset.xlsx', sheet_name='Data Dict')
            data.to_csv(DATA_DIR+file_name+'.csv', encoding='utf-8', index=False)
            data_dict.to_csv(DATA_DIR+file_name[:-5]+'_dict.csv', encoding='utf-8', index=False)
    elif dataset_num==2:
        file_name = 'dqlab_telco_final.csv'
        data = pd.read_csv(DATA_DIR + file_name)
    else:
        print('Load data Error: incorrect dataset number:', dataset_num)
        return pd.DataFrame()

    if DEBUG:
        print(f' Loaded {data.shape[0]} records.')
        # print(data.dtypes.to_string())
        print(pd.DataFrame(data.info()).to_string())
        # print(data_dict.dtypes)
        print(data.head())
        # print(data_dict.head())

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
    useless_columns = ['CustomerID', 'customerID', 'UpdatedAt']
    for col in useless_columns:
        if col in columns: # exists in df
            data.drop([col], axis=1, inplace = True)

    # 2. inspect categorical columns
    categorical = data.dtypes[data.dtypes=='object'].keys()
    if target in categorical:
        # specifics of DATASET_NUM==2, TARGET must be encoded to 0/1 before using OrdinalEncoder
        data.loc[data[target] ==  'No', target]=0
        data.loc[data[target] == 'Yes', target]=1
        data[target] = data[target].astype(int)
        # update categorical
        categorical = data.dtypes[data.dtypes=='object'].keys()
        if DEBUG:
            print(target,'encoded')

    if DEBUG and target in columns:
        print(f'\nCategorical columns: {list(categorical)}')
        # print distribution for each
        for col in categorical:
            print('\n by', data[col].value_counts().to_string())

        corr = data.corr(numeric_only=True)[target]
        print(f'\nCorrelation to {target}:\n{corr.to_string()}')

    # 3. inspect missing values
    nan_cols = data.columns[data.isnull().any()].to_list()
    if DEBUG:
            #list of columns with missing values and its percentage
            print(f'\nColumns with nulls:\n{nan_cols}')
            print_missing_values_table(data, na_name=True)

    # 4. fix missing values - fill with median values
    # fix_cols = ["Tenure","DaySinceLastOrder","OrderAmountHikeFromlastYear","OrderCount",
    #             "CouponUsed","HourSpendOnApp","WarehouseToHome",]
    data.loc[:,nan_cols] = data.loc[:,nan_cols].fillna(data.loc[:,nan_cols].median())

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

    print(f'\nFinal number of records:', data.shape[0],'/',total_rows_number,
                            '=', f'{data.shape[0]/total_rows_number*100:05.2f}%','\n')
    return data


def enc_save(enc, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(enc, f)

def enc_load(file_name):
    with open(file_name, 'rb') as f:
        enc = pickle.load(f)
        return enc
    return OrdinalEncoder()

def preprocess_data(df, ord_enc, dataset_num=DATASET_NUM, fit_enc=False):
    # if dataset_num==1:
    #     df = preprocess1(df)
    # elif dataset_num==2:
    #     df = preprocess2(df)
    # else:
    #   print('Preprocess data Error: incorrect dataset number:', dataset_num)
    #   return df

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
            enc_save(ord_enc, f'{DATA_DIR}encoder{dataset_num}.pkl')
            if DEBUG:
                print(' OrdinalEncoder categories:', ord_enc.categories_)
        else:
            # Only Transform the data (using pretrained encoder)
            df[categorical_features] = ord_enc.transform(df[categorical_features])

    return df


def load_data(dataset_num=DATASET_NUM):
    if dataset_num==1:
        df = load_dataset(1)
        ord_enc = OrdinalEncoder()
        df = preprocess_data(df, ord_enc, 1, fit_enc=True)
    elif dataset_num==2:
        df = load_dataset(2)
        ord_enc = OrdinalEncoder()
        df = preprocess_data(df, ord_enc, 2, fit_enc=True)
    else:
        print('Load data Error: incorrect dataset number:', dataset_num)
        return pd.DataFrame()

    return df

if __name__ == '__main__':
    # full dataset 1
    # df = load_data(1)
    # print('\nreal 1\n', df.head(5).to_string())
    df = load_data(2)
    print('\nreal 2\n', df.head(5).to_string())
    # exit()

    # mini dataset 1
    data = [
        (50001,1,4.0,'Mobile Phone',3,6.0,'Debit Card','Female',3.0,3,'Laptop & Accessory',2,'Single',9,1,11.0,1.0,1.0,5.0,159.93),
        (50002,1,None,'Phone',1,8.0,'UPI','Male',3.0,4,'Mobile',3,'Single',7,1,15.0,0.0,1.0,0.0,120.9),
        (50003,1,None,'Phone',1,30.0,'Debit Card','Male',2.0,4,'Mobile',3,'Single',6,1,14.0,0.0,1.0,3.0,120.28),
        (50004,1,0.0,'Phone',3,15.0,'Debit Card','Male',2.0,4,'Laptop & Accessory',5,'Single',8,0,23.0,0.0,1.0,3.0,134.07),
        (50005,1,0.0,'Phone',1,12.0,'CC','Male',None,3,'Mobile',5,'Single',3,0,11.0,1.0,1.0,3.0,129.6),
        (50006,1,0.0,'Computer',1,22.0,'Debit Card','Female',3.0,5,'Mobile Phone',5,'Single',2,1,22.0,4.0,6.0,7.0,139.19),
    ]

    columns = [
        'CustomerID','Churn','Tenure','PreferredLoginDevice','CityTier','WarehouseToHome','PreferredPaymentMode','Gender',
        'HourSpendOnApp','NumberOfDeviceRegistered','PreferedOrderCat','SatisfactionScore','MaritalStatus',
        'NumberOfAddress','Complain','OrderAmountHikeFromlastYear','CouponUsed','OrderCount',
        'DaySinceLastOrder','CashbackAmount'
    ]
    df = pd.DataFrame(data, columns=columns)
    df1 = preprocess_df(df)
    print('\n',1111,'\n', df1.head(1).to_string())
    dataset_num = 1
    ord_enc = enc_load(f'{DATA_DIR}encoder{dataset_num}.pkl')
    df = preprocess_data(df1, ord_enc, 1, fit_enc=False)
    print('\n',1111-2,'\n', df.head(1).to_string())

    # exit()

    # dataset 2
    data = [
        (202006,45759018157,'Female','No','Yes',1,'No','No','Yes','Yes',29.85,29.85,'No'),
        (202006,45315483266,'Male','No','Yes',60,'Yes','No','No','Yes',20.5,1198.8,'No'),
        (202006,45236961615,'Male','No','No',5,'Yes','Yes','Yes','No',104.1,541.9,'Yes'),
        (202006,45929827382,'Female','No','Yes',72,'Yes','Yes','Yes','Yes',115.5,8312.75,'No'),
        (202006,45305082233,'Female','No','Yes',56,'Yes','Yes','Yes','No',81.25,4620.4,'No'),
    ]

    columns = [
        'UpdatedAt','customerID','gender','SeniorCitizen','Partner','tenure','PhoneService','StreamingTV',
        'InternetService','PaperlessBilling','MonthlyCharges','TotalCharges','Churn',
    ]
    df = pd.DataFrame(data, columns=columns)
    df = preprocess_df(df)
    print('\n',2222,'\n', df.head(1).to_string())
    dataset_num = 2
    ord_enc = enc_load(f'{DATA_DIR}encoder{dataset_num}.pkl')
    df = preprocess_data(df1, ord_enc, 1, fit_enc=False)
    print('\n',2222-2,'\n', df.head(1).to_string())
