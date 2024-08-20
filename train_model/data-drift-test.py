import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # Disabling false positive warning
import whylogs as why
from predict import predict_df
from preprocess import load_data, enc_load, preprocess_data, model_dir
from pymongo import MongoClient
from sklearn.preprocessing import OrdinalEncoder

from settings import DEBUG, DATA_DIR, DATASET_NUM, MODEL_DIR, TARGET  # isort:skip

DEBUG = True  # True # False # override global settings

# MONGODB_ADDRESS = 'mongodb://mongo:27017' #127.0.0.1
MONGODB_ADDRESS = 'mongodb://172.27.0.3:27017'

# As a drift test, let's compare
# - target - recorded web service requests with predictions
# - reference - part of original dataset with predictions


def load_data_mongodb(mongodb_address=MONGODB_ADDRESS):
    try:
        mongo_client = MongoClient(mongodb_address)
        db = mongo_client.get_database("prediction_service")
        collection = db.get_collection("data")

        print('\n\nRequest MONGODB directly')
        df = pd.DataFrame(list(collection.find()))
        df.drop(['_id'], axis=1, inplace=True)  # , 'CustomerID'
        print('MONGODB rows num', df.shape[0])
        if DEBUG:
            df.info()
            print(df.head(1).to_string())
            print(df.head(1).to_dict('records')[0])

            for col in ['Gender']:  # , 'CouponUsed'
                print('\n by', df[col].value_counts().to_string())

            print('\n Tenure describe:\n', df['Tenure'].describe())
        return df
    except Exception as e:
        print(e)
        return pd.DataFrame()


def load_data_orig(dataset_num=DATASET_NUM, num=300):
    df = load_data().head(num)
    y_pred = predict_df(df, MODEL_DIR)
    df['prediction'] = y_pred
    if DEBUG:
        last_doc1 = df.head(1).to_dict('records')[0]
        print('last_doc1:', last_doc1)
    return df


def mutate_df(df_target, mutate_col):
    print('\n before', df_target[mutate_col].value_counts().to_string())
    # changing values of column with low feature importance
    df_target[mutate_col] = np.random.choice(
        [0, 0, 0, 1], df_target.shape[0]
    )  # change distribution - more 0
    print('\n after', df_target[mutate_col].value_counts().to_string())
    return df_target


def analyze_drift(target_view, reference_view, verbose=DEBUG):
    from whylogs.viz.drift.column_drift_algorithms import calculate_drift_scores

    scores = calculate_drift_scores(
        target_view=target_view, reference_view=reference_view, with_thresholds=True
    )
    if verbose:
        print('\nDrift scores:', scores)
    drift_scores = {}
    for key, feature_score in scores.items():
        if feature_score == None or (not 'drift_category' in feature_score):
            print('error?', key)
            continue
        elif feature_score['drift_category'] == 'NO_DRIFT':
            continue
        if verbose:
            print(' ', key, feature_score['drift_category'])
        drift_scores.update({key: feature_score['drift_category']})
    if verbose and len(drift_scores) == 0:
        print(' no drifts detected.')
    return drift_scores


df_recorded = load_data_mongodb()
if df_recorded.shape[0] == 0:
    exit()

# y = df.pop('prediction') #'prediction'

df_orig = load_data_orig()

df_target = df_recorded.drop([TARGET], axis=1)
df_reference = df_orig.drop([TARGET], axis=1)


results = why.log(df_reference)
print('\nresults:', results)


result_target = why.log(pandas=df_target)
prof_view_target = result_target.view()
print('\nprof_view_target:\n', prof_view_target.to_pandas())

result_ref = why.log(pandas=df_reference)
prof_view_ref = result_ref.view()
print('\nprof_view_ref:\n', prof_view_ref.to_pandas())

drift_scores = analyze_drift(
    target_view=prof_view_target, reference_view=prof_view_ref, verbose=True
)
print('\nDrift scores:', drift_scores)
# as target was actually a part of original dataset -  no drifts detected

# Now let's 'mutate' a feature with low importance - gender, and see the difference
mutate_col = 'Gender'
print('Drift experiment:', mutate_col)
df_target1 = mutate_df(df_reference, mutate_col)

# we mutate reference - from original dataset
df1 = load_data().head(300)
df1 = mutate_df(df1, mutate_col)
y_pred = predict_df(df1, MODEL_DIR)
df1['prediction'] = y_pred
if DEBUG:
    last_doc1 = df1.head(1).to_dict('records')[0]
    print('last_doc1:', last_doc1)

df_reference1 = df1.drop([TARGET], axis=1)

result_ref1 = why.log(pandas=df_reference1)
prof_view_ref1 = result_ref1.view()
print('\nprof_view_ref1:\n', prof_view_ref1.to_pandas())

drift_scores1 = analyze_drift(
    target_view=prof_view_target, reference_view=prof_view_ref1, verbose=True
)
print('\nDrift scores1:', drift_scores1)
# so as we designed, we have Gender DRIFT
# BUT as gender has low importance for prediction -  no drift in prediction detected!


# visualizations
import os

DATA_DIR = os.getcwd() + '/'

from whylogs.viz import NotebookProfileVisualizer

visualization = NotebookProfileVisualizer()
visualization.set_profiles(
    target_profile_view=prof_view_target, reference_profile_view=prof_view_ref
)

visualization.write(
    rendered_html=visualization.profile_summary(),
    html_file_name=DATA_DIR + "example1",
)

visualization.write(
    rendered_html=visualization.summary_drift_report(),
    html_file_name=DATA_DIR + "example2",
)

visualization.write(
    rendered_html=visualization.double_histogram(
        feature_name=["CashbackAmount", "SatisfactionScore"]
    ),
    html_file_name=DATA_DIR + "example3",
)

visualization.write(
    rendered_html=visualization.difference_distribution_chart(feature_name="prediction"),
    html_file_name=DATA_DIR + "example4",
)
