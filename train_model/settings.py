# show extra information for checking execution
DEBUG = False  # True # False

DATASET_NUM = 1  # 1 # 2
DATA_DIR = f'./data/'  # ! with '/' at the end!
MODEL_DIR = f'./model/{DATASET_NUM}/'  # ! with '/' at the end!
MODEL_PREFIX = MODEL_DIR.lstrip('./')  # so S3 prefix would be 'model/1/' or 'model/2/'

VISUALS_DIR = './screenshots/'  # ! with '/' at the end!

# preprocessing, training, prediction
TARGET = "Churn"  # labels column

# preprocessing
REMOVE_OUTLIERS = False  # True # False

# mlflow training
EXPERIMENT_NAME = "Training Regression model for churn prediction"
HPO_EXPERIMENT_NAME = "Testing trained Regression models"


# web app settings
PORT = 5555
NEW_API_PARAM_NUM = 13


# from settings import DEBUG, DATASET_NUM, DATA_DIR, MODEL_DIR,
# from settings import REMOVE_OUTLIERS, TARGET
# from settings import EXPERIMENT_NAME
