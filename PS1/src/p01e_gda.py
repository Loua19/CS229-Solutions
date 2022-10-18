import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***

    # NOTE: GDA performs worse than logistic regression on ds1 because the classes are not distributed normally.

    clf = GDA()
    clf.fit(x_train, y_train)

    util.plot(x_train, y_train, clf.theta, '../output/p01gGDA_{}.png'.format(pred_path[-5]))

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***

        # m = #data, n = dimension of data (x)
        m, n = x.shape
        
        # Calculating parameters
        phi = 1/m * np.sum(y==1)
        mu_0 = np.sum(x[y==0], axis = 0) / np.sum(y==0)
        mu_1 = np.sum(x[y==1], axis = 0) / np.sum(y==1)

        # Calculating sigma
        sigma = np.copy(x)
        sigma[y==0]+=-mu_0
        sigma[y==1]+=-mu_1
        sigma = 1/m * np.matmul(sigma.T, sigma)

        # Calculating theta
        sigmaInv = np.linalg.inv(sigma)
        self.theta = np.matmul(sigmaInv, mu_1 - mu_0)
        theta_0 = 0.5 * np.dot((mu_0 + mu_1), np.matmul(sigmaInv, (mu_0 - mu_1))) - np.log((1-phi)/phi)
        self.theta = np.append(theta_0, self.theta)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        return 1/(1 + np.exp(-( np.dot(x, self.theta))))

        # *** END CODE HERE

if __name__ == "__main__":
    
    # For testing purposes
    main(train_path='../data/ds2_train.csv',
         eval_path='../data/ds1_valid.csv',
         pred_path='../output/')