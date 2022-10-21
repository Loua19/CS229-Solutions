import matplotlib.pyplot as plt
import numpy as np
import util
from p05b_lwr import main as p05b

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data

    mse = {}
    
    for tau in tau_values:

        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_valid)

        plt.figure()
        plt.plot(x_valid[:, 1], y_valid, 'bo', linewidth=2, label='True value')
        plt.plot(x_valid[:, 1], y_pred, 'ro', linewidth=2, label='Model prediction')
        plt.legend(loc='upper left')
        plt.suptitle('Model with tau = ' + str(tau), fontsize=12)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('../output/p05b_' + str(tau) + '.png')

        m, n = x_valid.shape
        mse[tau] = 1/m * np.sum((y_pred - y_valid)**2)

    return(mse)
 
    # *** END CODE HERE ***

if __name__ == "__main__":

    # For testing purposes
    print(main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
        train_path ='../data/ds5_train.csv',
        valid_path = '../data/ds5_valid.csv',
        test_path = '../data/ds5_valid.csv',
        pred_path = '../output'))