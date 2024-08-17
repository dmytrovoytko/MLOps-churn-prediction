import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from predict import print_results

from settings import DEBUG, TARGET # isort:skip
DEBUG = True # True # False # override global settings

def prepare_training(df, params):
    # dataset in df is already preprocessed - cleaned, OrdinalEncoder applied
    # determine test_size
    test_size = params.get('test_size', 0.2)
    train_size = 1 - test_size
    # target labels column
    target = TARGET
    cols = df.columns.to_list()

    random_state = params.get('random_state', 42)
    if params.get('balance', False): # balance dataset on target labels?
        print(f'\nBalancing dataset...')
        # as Churn 0 / 1 are unbalanced (~ 3:1 or more), let's mix a good proportion manually
        dataset0 = list(df[df[target]==0].sample(frac=1, random_state=random_state).itertuples(index=False, name=None))
        dataset1 = list(df[df[target]==1].sample(frac=1, random_state=random_state).itertuples(index=False, name=None))

        # minimal subset to have equal number of target labels
        min_len = min(len(dataset0), len(dataset1))
        params['min_subset'] = min_len
        train_data = dataset0[:int(min_len*train_size)] + dataset1[:int(min_len*train_size)]
        test_data = dataset0[int(min_len*train_size):int(min_len*1.00)+1] + dataset1[int(min_len*train_size):int(min_len*1.00)+1]

        X_train = pd.DataFrame(train_data, columns=cols)
        X_test = pd.DataFrame(test_data, columns=cols)
        y_train = X_train.pop(target)
        y_test = X_test.pop(target)
        if DEBUG:
            print(' by', df[target].value_counts().to_string())
            print(f"\nTotal: {df.shape[0]} // Min subset: {min_len} // Split: train {len(train_data)} + test {len(test_data)}")
            print(f"\nX_train: {X_train.shape[0]}\n{X_train.head()}\n\n")
            print(' by', y_train.value_counts().to_string())
    else:
        cols.remove(target)
        X_train, X_test, y_train, y_test = train_test_split(df[cols], df[target], test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, params


def train_model(df, params, random_state=42):

        X_train, X_test, y_train, y_test, _ = prepare_training(df,
                                                        {'balance': params['balance'],
                                                            'test_size': params['test_size'],
                                                            'random_state': random_state
                                                        })
        classifier_name = params['classifier']
        print(f"\nGridSearchCV {classifier_name}...")
        # Define the parameter grid to tune the hyperparameters (for gbtree)
        if classifier_name=='XGBClassifier':
            param_grid = {
                'eta': [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0,4],  # ~ learning_rate
                'gamma': [0, 1, 2, 3], # min_split_loss The larger gamma is, the more conservative the algorithm will be
                'max_depth': [2, 3, 4, 5, 6],
                'min_child_weight': [0, 1, 2],
                'max_delta_step': [0, 1, 2, 4],
                'lambda': [0, 1, 2], # reg_lambda L2 regularization term on weights (analogous to Ridge regression)
            }
            classifier = xgb.XGBClassifier(booster='gbtree', random_state=random_state)
        elif classifier_name=='DecisionTreeClassifier':
            param_grid = {
                'max_depth': [3, 5, 7, 10, 20, 30, None],
                'min_samples_leaf': [1, 3, 5],
                'min_samples_split': [2, 6, 10],
            }
            classifier = DecisionTreeClassifier(random_state=random_state) # DecisionTreeRegressor(random_state=random_state) # Initialize a decision tree regressor
        elif classifier_name=='RandomForestClassifier':
            param_grid = {
                'n_estimators': [25, 50, 100, 150, 200],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [3, 6, 9, None],
                'min_samples_leaf': [1, 3, 5],
                'min_samples_split': [2, 6, 10],
                'max_leaf_nodes': [3, 6, 9],
            }
            classifier = RandomForestClassifier(random_state=random_state)
        else:
            return

        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid,
                                   cv=params['cv'],
                                   n_jobs=-1,
                                   verbose=1, #3, #2,
                                   scoring=params['estimator'] #'balanced_accuracy', #'accuracy' #'neg_mean_squared_error'
                                   )
        grid_search.fit(X_train, y_train)
        best_classifier = grid_search.best_estimator_ # Get the best estimator from the grid search
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f'\nHPO grid search: {grid_search.scorer_}, {grid_search.n_splits_} splits')
        print(f"Best parameters: {best_params}")
        if DEBUG:
            print(f" best_score_: {best_score}")
            features_ = list(df.columns)
            features_.remove(TARGET)
            # print(f" feature_importances_: {grid_search.best_estimator_.feature_importances_}")
            # print(f" features_: {features_}")
            feature_importances_ = dict(zip(features_, list(grid_search.best_estimator_.feature_importances_)))
            print(classifier_name, 'feature_importances_', sorted(feature_importances_.items(), key=lambda x:x[1], reverse=True))
            # print(f" cv_results_: {grid_search.cv_results_}") # huge set of data

        # calculating test prediction metrics
        y_pred = best_classifier.predict(X_test)
        print_results(f"{classifier_name}, optimized for {params['estimator']}", y_test, y_pred, verbose=True)
        key_metric1 = accuracy_score(y_test, y_pred)
        key_metric2 = balanced_accuracy_score(y_test, y_pred)

        return best_classifier, best_params, key_metric1, key_metric2


if __name__ == '__main__':
    # to test model training without orchestration and HPO
    from time import time

    from preprocess import load_data
    df = load_data()
    estimator = 'accuracy'
    for classifier in [
                    'DecisionTreeClassifier',
                    # 'RandomForestClassifier',
                    # 'XGBClassifier',
                    ]:
        t_start = time()
        params = {'classifier': classifier,
                    'estimator': estimator,
                    'cv': 2,
                    'test_size': 0.2,
                    'balance': False,
                    }
        best_classifier, best_params, key_metric1, key_metric2 = train_model(df, params, random_state=42)
        print(f' Finished in {(time() - t_start):.3f} second(s)\n')
