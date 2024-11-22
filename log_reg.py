import numpy as np
import pandas as pd


def get_cancer_np_array():
    cancer = pd.read_csv('input_data\cancer.csv').to_numpy()
    return cancer

def get_validation_np_array():
    validation = pd.read_json('input_data\cancer_cv.json').to_numpy()
    return validation

def get_lambdas(start=1e-10, finish=1e-2, length=1000):
    return np.linspace(start, finish, length)

def newtons_method(x: np.ndarray, d_k: np.ndarray, theta_k: np.ndarray, y: np.ndarray, pi_k: np.ndarray):
    return np.linalg.pinv(x.T @ d_k @ x) @ x.T @ (d_k @ x @ theta_k + y - pi_k)

def cross_entropy(y_true_array: np.ndarray, y_pred_array: np.ndarray):
    sum = 0
    n = y_pred_array.shape[0]
    for y_true, y_pred in zip(y_true_array, y_pred_array):
        sum += y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)        
    return -1 * sum / n

def likelihoood(probabilities: np.ndarray):
    return np.prod(probabilities)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(input, validation, theta_initial):
    input_rows_num = input.shape[1]
    if theta_initial is None:
        local_theta_init = np.zeros(input_rows_num)
    else:
        local_theta_init = np.copy(theta_initial)

    

if __name__ == "__main__":
   cancer = get_cancer_np_array() 
   validation  = get_validation_np_array() 
   lambdas = get_lambdas()
   

