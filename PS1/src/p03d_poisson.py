import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path

    # NOTE: The problem here that took hours to debug was find a working value of lr.

    model = PoissonRegression()
    model.step_size = lr
    model.fit(x_train, y_train)

    # Plotting the prediction

    y_pred = model.predict(x_eval)
    plt.plot(y_eval, 'go', label='True value')
    plt.plot(y_pred, 'rx', label='Model prediction')
    plt.suptitle('Validation Set', fontsize=12)
    plt.legend(loc='upper left')
    plt.savefig('../output/p03d.png')
    
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        m, n = x.shape
        self.theta = np.zeros(n)
        j = 0

        # Implementing (batch) gradient descent

        while True:

            theta = np.copy(self.theta)
            self.theta += self.step_size * (x.T).dot(y - np.exp( x.dot(self.theta)))

            if np.linalg.norm(self.theta - theta) < self.eps or j > self.max_iter:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***

        return np.exp(np.matmul(x, self.theta))

        # *** END CODE HERE ***

if __name__ == "__main__":
    
    # For testing purposes
    main(lr = 6e-11,
         train_path='../data/ds4_train.csv',
         eval_path='../data/ds4_valid.csv',
         pred_path='../output/')