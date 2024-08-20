import warnings

# warnings.simplefilter("ignore", category=UserWarning)
warnings.filterwarnings('ignore')

from time import time

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow, task
from settings import DATA_DIR, DATASET_NUM, EXPERIMENT_NAME, MODEL_DIR, MODEL_PREFIX, TARGET

from settings import DEBUG  # isort:skip

DEBUG = True  # True # False # override global settings

from utils import S3_ENDPOINT_URL, S3_BUCKET  # isort:skip

from predict import predict_df, print_results
from preprocess import load_data

from train_model import train_model

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

        best_classifier, best_params, key_metric1, key_metric2 = train_model(
            df, params, random_state
        )

        mlflow.log_params(best_params)
        mlflow.log_metric('test_accuracy_score', key_metric1)
        mlflow.log_metric('test_balanced_accuracy_score', key_metric2)
        mlflow.log_metric('duration', (time() - t_start))
        print('model_uri:', f"runs:/{run.info.run_id}/model")
        # print(f"Model saved in run {mlflow.active_run().info.run_uuid}\n")

    # mlflow.end_run()
    print(f'Experiment finished in {(time() - t_start):.3f} second(s)\n')


def feature_importances_chart(feature_importances, classifier_name):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from settings import VISUALS_DIR

    df = (
        pd.DataFrame(feature_importances, columns=['Feature', 'Importance'])
        .set_index('Feature')
        .sort_values("Importance", ascending=False)
    )
    ax = sns.barplot(x="Importance", y="Feature", data=df, color="b")
    title = f'Model {DATASET_NUM} {classifier_name} feature importances'
    # ax = df.plot(kind='barh', legend=False) # or .barh() # looks worse than by sns
    plt.title(title)
    plt.tight_layout()  # proper padding for feature names
    plt.savefig(f'{VISUALS_DIR}bar-feature_importances_{classifier_name}-{DATASET_NUM}.png')


def model_features(MODEL_DIR, verbose=DEBUG):
    try:
        model = mlflow.pyfunc.load_model(MODEL_DIR)  # f"runs:/{run_id}/model")
        print('\n_model_meta._signature:\n', model._model_meta._signature)
        features_ = model._model_meta._signature.inputs.input_names()
        import pickle

        model = pickle.load(open(f'{MODEL_DIR}model.pkl', 'rb'))
        feature_importances_ = dict(
            zip(features_, list(model.best_estimator_.feature_importances_))
        )
        feature_importances = sorted(feature_importances_.items(), key=lambda x: x[1], reverse=True)
        classifier_name = str(type(model.best_estimator_)).strip('<\'>').split('.')[-1]
        print(classifier_name, 'feature_importances:', feature_importances)

        feature_importances_chart(feature_importances, classifier_name)

    except Exception as e:
        print('!!! Exception while loading model:', e)
        return

    if 'XGBClassifier' in classifier_name:
        try:
            import matplotlib.pyplot as plt
            from xgboost import plot_importance

            plot_importance(model.best_estimator_._Booster)
            # plt.show()
            VISUALS_DIR = './data/'
            plt.tight_layout()  # proper padding for feature names
            plt.savefig(VISUALS_DIR + f'feature_importances_{classifier_name}-{DATASET_NUM}.png')
        except Exception as e:
            print('Error:', e)


@task
def run_register_model(MODEL_DIR, top_n: int = 1):
    print('\nRegistering best model...')
    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.training_accuracy_score DESC"],
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
        max_results=1,  # 3 to see and compare
        order_by=[f"metrics.{KEY_METRIC} DESC"],
    )
    if len(best_runs) == 0:
        print(f'\n!!! Error! No runs for this dataset ({DATASET_NUM}) found!')
        return

    best_run = best_runs[0]
    print(f'\nBest results: run_id {best_run.info.run_id}, metrics {best_run.data.metrics}')
    print(f'params: {best_run.data.params}')

    # Register the best model TODO get last model version? compare, register only if better
    result = mlflow.register_model(f'runs:/{best_run.info.run_id}/model', "churn-prediction")
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

    if S3_ENDPOINT_URL:
        try:
            from utils import S3 as s3
            from utils import s3_list_buckets, s3_list_objects, s3_upload_files

            buckets = s3_list_buckets(s3)
            bucket_name = S3_BUCKET
            if not bucket_name in buckets:
                s3.create_bucket(Bucket=bucket_name)
                buckets = s3_list_buckets(s3)
                print(S3_BUCKET, 'created?', buckets)
            # upload model dir & encoder file to S3 BUCKET
            s3_upload_files(s3, bucket_name, MODEL_DIR, prefix_key=MODEL_PREFIX)
            s3_upload_files(s3, bucket_name, f'{MODEL_DIR}encoder.pkl', prefix_key=MODEL_PREFIX)
            if DEBUG:
                objects = s3_list_objects(s3, bucket_name, filter='')
                print('\n--------')
                print('S3 objects:', sorted(objects))
                print('--------')
        except Exception as e:
            print('!!! Uploading model to S3 failed:', e)

    # run_id = best_run.info.run_id
    model_features(MODEL_DIR, verbose=DEBUG)


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
        for estimator in ['accuracy', 'balanced_accuracy', 'roc_auc']:  # _score
            for cv in [2]:  # [2, 5]: # (x)-fold cross validation
                for balance in [False]:  # [False, True]: # balance dataset on target labels?
                    params = {
                        'classifier': classifier,
                        'estimator': estimator,
                        'dataset': DATASET_NUM,
                        'cv': cv,
                        'test_size': 0.2,
                        'balance': balance,
                    }
                    run_experiment(
                        df,
                        params=params,
                        run_name=f"classifier {params['classifier']}, estimator {params['estimator']}",
                    )

    # register and save the best model
    run_register_model(MODEL_DIR)

    # Testing model on FULL dataset
    test_model(df, MODEL_DIR)


if __name__ == '__main__':
    # run Prefect workflow
    ml_workflow()

    # quick independent testing
    TESTING_MODE1 = False  # True # False
    if TESTING_MODE1:
        # load and preprocess dataset
        df = load_data()
        test_model(df, MODEL_DIR)

    REGISTER_MODEL = False  # False # True
    if REGISTER_MODEL:
        MODEL_DIR = f'./model/{DATASET_NUM}/'
        run_register_model(MODEL_DIR)

    # model_features(MODEL_DIR, verbose=DEBUG)

    TESTING_S3 = False  # True # False
    if TESTING_S3 and S3_ENDPOINT_URL:
        from utils import S3 as s3
        from utils import s3_list_buckets, s3_list_objects, s3_upload_files

        buckets = s3_list_buckets(s3)
        bucket_name = S3_BUCKET
        if not bucket_name in buckets:
            s3.create_bucket(Bucket=bucket_name)
            buckets = s3_list_buckets(s3)
            print(S3_BUCKET, 'created?', buckets)
        objects = s3_list_objects(s3, bucket_name, filter='model/')
        print('\n--------')
        print('S3 objects:', sorted(objects))
        print('--------')
        s3_upload_files(s3, bucket_name, MODEL_DIR, prefix_key=MODEL_PREFIX)
        s3_upload_files(s3, bucket_name, f'{MODEL_DIR}encoder.pkl', prefix_key=MODEL_PREFIX)
        objects = s3_list_objects(s3, bucket_name, filter='')
        print('\n--------')
        print('S3 objects:', sorted(objects))
        print('--------')
