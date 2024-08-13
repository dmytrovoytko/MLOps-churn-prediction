# show extra information for checking execution
DEBUG = False # True # False

DATASET_NUM = 1 # 1 # 2
DATA_DIR = f'./data/' # ! with '/' at the end!
MODEL_DIR = f'./model/{DATASET_NUM}/'  # ! with '/' at the end!

# preprocessing
REMOVE_OUTLIERS = False # True # False

# preprocessing, training, prediction
TARGET = "Churn" # labels column 

# mlflow training
EXPERIMENT_NAME = "Training Regression model for churn prediction"
HPO_EXPERIMENT_NAME = "Testing trained Regression models"

# from settings import DEBUG, DATASET_NUM, DATA_DIR, MODEL_DIR, 
# from settings import REMOVE_OUTLIERS, TARGET
# from settings import EXPERIMENT_NAME