import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # NOTE: The reason that the model is bad for 5b is because tau is too large

    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_valid)

    # Plotting data
    plt.figure()
    plt.plot(x_valid[:, 1], y_valid, 'bo', linewidth=2, label='True value')
    plt.plot(x_valid[:, 1], y_pred, 'ro', linewidth=2, label='Model prediction')
    plt.legend(loc='upper left')
    plt.suptitle('Model with tau = ' + str(tau), fontsize=12)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('../output/p05b.png')
    
    # Reporting MSE
    m, n = x_valid.shape
    mse = 1/m * np.sum((y_pred - y_valid)**2)
    return(mse)

    # *** END CODE HERE *** 


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***

        self.x = x
        self.y = y

        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        m, n = x.shape
        theta = np.zeros((m,n))

        # Initialising W, solving normal equations and returning theta for each data point
        for i in range(m):
            W =  np.diag( np.exp(  (np.sum( (self.x - x[i])**2, axis = 1)) / (-2*self.tau**2) ) )
            theta[i] = np.linalg.inv( ((self.x).T) @ W @ self.x ) @ ( ((self.x).T) @ W ) @ self.y

        return np.sum( x * theta, axis = 1)

        # *** END CODE HERE ***

if __name__ == "__main__":

    # For testing purposes
    print(main(tau = 0.5,
         train_path='../data/ds5_train.csv',
         eval_path='../data/ds5_valid.csv'))