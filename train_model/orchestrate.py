import warnings # supress warnings
# warnings.simplefilter("ignore", category=UserWarning)
warnings.filterwarnings('ignore')

from time import time

import numpy as np
import pandas as pd
# Display all columns
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


import seaborn as sns
import matplotlib.pyplot as plt

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from prefect import flow, task

from settings import DEBUG, DATASET_NUM, DATA_DIR, TARGET
from settings import MODEL_DIR, EXPERIMENT_NAME

from preprocess import load_data
from train_model import train_model
from predict import predict_df, print_results


###########################

@task
def run_experiment(df, params, run_name):
    t_start = time()
    random_state = 42
    print(f'\nStarting experiment [{run_name}]...')
    print(f' params: {params}')
    
    mlflow.autolog()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)

        best_classifier, best_params, key_metric1, key_metric2 = train_model(df, params, random_state)

        mlflow.log_params(best_params)
        mlflow.log_metric('test_accuracy_score', key_metric1)
        mlflow.log_metric('test_balanced_accuracy_score', key_metric2)
        mlflow.log_metric('duration', (time() - t_start))
        print('model_uri:', f"runs:/{run.info.run_id}/model")
        # print(f"Model saved in run {mlflow.active_run().info.run_uuid}\n")

    # mlflow.end_run()
    print(f'Experiment finished in {(time() - t_start):.3f} second(s)\n')

@task
def run_register_model(MODEL_DIR, top_n: int =1):
    print('\nRegistering best model...')
    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.training_accuracy_score DESC"]
    )
    for run in runs:
        print(f'Best performing models from run_id {run.info.run_id}, {run.data.params}')
        # test model again?

    # Select the model with the highest key metric
    KEY_METRIC = 'test_accuracy_score'
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_runs = client.search_runs( 
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        filter_string=f'params.dataset="{DATASET_NUM}"',
        max_results=1, # 3 to see and compare
        order_by=[f"metrics.{KEY_METRIC} DESC"]
    )
    if len(best_runs)==0:
        print(f'\n!!! Error! No runs for this dataset ({DATASET_NUM}) found!')
        return

    best_run = best_runs[0]
    print(f'\nBest results: run_id {best_run.info.run_id}, metrics {best_run.data.metrics}\nparams: {best_run.data.params}')

    # Register the best model TODO get last model version? compare, register only if better
    result = mlflow.register_model(
        f'runs:/{best_run.info.run_id}/model', "churn-prediction"
    )
    print(f'\nModel registered: {result}')
    print(f' Name: {result.name} Version: {result.version} run_id: {result.run_id}')

    # model_uri = f"models:/{model_name}/{model_version}"
    # model = mlflow.sklearn.load_model(model_uri)
    # nlp = mlflow.spacy.load_model(f'runs:/{best_run.info.run_id}/model') # -best
    # if DEBUG:
    #     print(nlp.config)

    # copy best model to ./model dir
    import shutil
    src = result.source.replace('file:/', '')
    dest = MODEL_DIR
    shutil.copytree(src, dest, dirs_exist_ok=True)  # 3.8+ only!
    print(f'\nModel saved to {dest}')

@task
def test_model(test_data, MODEL_DIR):
    print(f'\nTesting model {MODEL_DIR}')
    y_pred = predict_df(test_data, MODEL_DIR, verbose=DEBUG)
    print_results('Saved model', test_data[TARGET], y_pred, verbose=True)

###########################

@flow
def ml_workflow():
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    df = load_data()

    # Training with different hyper parameters
    for classifier in [
                    'DecisionTreeClassifier', 
                    # 'RandomForestClassifier', 
                    # 'XGBClassifier',
                    ]:
        for estimator in ['accuracy', 'balanced_accuracy', 'roc_auc']: #_score
            for cv in [2]: # [2, 5]: # (x)-fold cross validation
                for balance in [False]: # [False, True]: # balance dataset on target labels? 
                    params = {'classifier': classifier, 
                                'estimator': estimator,
                                'dataset': DATASET_NUM,
                                'cv': cv,
                                'test_size': 0.2,
                                'balance': balance, 
                                }
                    run_experiment(df, params=params, 
                                    run_name=f"classifier {params['classifier']}, estimator {params['estimator']}")

    # register and save the best model
    run_register_model(MODEL_DIR)

    # Testing model on FULL dataset
    test_model(df, MODEL_DIR)


if __name__ == '__main__':
    # run Prefect workflow
    ml_workflow()

    # independent testing
    TESTING_MODE1 = False # True # False
    if TESTING_MODE1:
        # load and preprocess dataset
        df = load_data() 
        test_model(df, MODEL_DIR)

    REGISTER_MODEL = False # False # True
    if REGISTER_MODEL:
        MODEL_DIR = f'./model/{DATASET_NUM}/'
        run_register_model(MODEL_DIR)

