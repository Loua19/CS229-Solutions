import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***

    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c

    # Importing data and training model using x1, x2, t: 

    x_train, t_train = util.load_dataset(train_path, 't', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, 't', add_intercept=True)

    xtModel = LogisticRegression()
    xtModel.fit(x_train, t_train)
    util.plot(x_test, t_test, xtModel.theta, '../output/p02c_{}.png'.format(pred_path[-5]))

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d

    x_train, y_train = util.load_dataset(train_path, 'y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, 'y', add_intercept=True)

    xyModel = LogisticRegression()
    xyModel.fit(x_train, y_train)
    util.plot(x_test, y_test, xyModel.theta, '../output/p02d_{}.png'.format(pred_path[-5]))

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e

    x_valid, y_valid = util.load_dataset(valid_path, 'y', add_intercept=True)

    # Calculating alpha and adjusting theta[0]
    alpha = 1/(sum(y_valid==1)) * np.sum(xyModel.predict(x_valid[y_valid==1]))
    xyModel.theta[0] = xyModel.theta[0] + np.log((2/alpha) -1)
    util.plot(x_test, t_test, xyModel.theta, '../output/p02e_{}.png'.format(pred_path[-5]))

    # *** END CODER HERE

if __name__ == "__main__":
    
    # For testing purposes
    main(train_path='../data/ds3_train.csv',
         valid_path='../data/ds3_valid.csv',
         test_path='../data/ds3_test.csv',
         pred_path='../output/')