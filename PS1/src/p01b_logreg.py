import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # Fitting the data
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    #plotting the solution using the util method
    util.plot(x_train, y_train, clf.theta, '../output/p01gLogReg_{}.png'.format(pred_path[-5]))

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        # TODO:
        # - Redo this without using column vectors (i.e for )

        # m = number of data points, n = number of features 
        m, n = x.shape
        y = np.reshape(y, (m,1))
        self.theta = np.zeros((n, 1))

        i = 0
        while True:

            thetaPrev = np.copy(self.theta)

            # Vectorised calculations:
            h_x = 1/(1 + np.exp(-np.matmul(x, self.theta)))
            H = 1/m * np.matmul(((h_x*(1-h_x)).T*(x.T)), x)
            gradL = (-1/m) *np.matmul(x.T, (y - h_x))
            self.theta = self.theta - np.matmul(np.linalg.inv(H), gradL)

            # eps and max_iter come from LinearModel class which LogisticRegression extends
            if np.linalg.norm(self.theta - thetaPrev) <= self.eps or i >= self.max_iter:
                break
            
            i = i + 1

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        return 1/(1 + np.exp(-np.matmul(x, self.theta)))

        # *** END CODE HERE ***

if __name__ == "__main__":
    
    # For testing purposes
    main(train_path='../data/ds2_train.csv',
         eval_path='../data/ds1_valid.csv',
         pred_path='../output/')