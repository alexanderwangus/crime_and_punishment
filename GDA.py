import numpy as np
import utils
import math

from sklearn import linear_model


RACE_BLIND = False
GDA = True

def main(train_path, eval_path, pred_path):
    x_train, y_train = utils.get_data(utils.TRAIN_PATH, race_blind = RACE_BLIND)
    x_validate, y_validate = utils.get_data(utils.VALIDATE_PATH, race_blind = RACE_BLIND)
    x_test, y_test = utils.get_data(utils.TEST_PATH, race_blind = RACE_BLIND)
    feature_names = utils.get_feature_names(utils.TRAIN_PATH, race_blind = RACE_BLIND)

    #To train a GDA model
    clf = GDA()
    clf.fit(x_train, y_train)

    # Plot decision boundary on validation set
    plot_path = pred_path.replace('.txt', '.eps')
    
    util.plot(x_validate, y_validate, clf.theta, plot_path)
    x_validate = util.add_intercept(x_validate)

    # Use np.savetxt to save outputs from validation set to pred_path
    p_validate = clf.predict(x_validate)
    np.savetxt(pred_path, p_eval)

    print("Train Accuracy: ", evaluate(X_train, y_train, reg))
    print("Validation Accuracy: ", evaluate(X_validate, y_validate, reg))

class GDA(linear_model):
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
        m, n = x.shape

        # Find phi, mu_0, mu_1, and sigma
        phi = 1 / m * np.sum(y == 1)
        mu_0 = (y == 0).dot(x) / np.sum(y == 0)
        mu_1 = (y == 1).dot(x) / np.sum(y == 1)
        mu_yi = np.where(np.expand_dims(y == 0, -1),
                         np.expand_dims(mu_0, 0),
                         np.expand_dims(mu_1, 0))
        sigma = 1 / m * (x - mu_yi).T.dot(x - mu_yi)

        # Write theta in terms of the parameters
        self.theta = np.zeros(n + 1)
        sigma_inv = np.linalg.inv(sigma)
        mu_diff = mu_0.T.dot(sigma_inv).dot(mu_0) \
            - mu_1.T.dot(sigma_inv).dot(mu_1)
        self.theta[0] = 1 / 2 * mu_diff - np.log((1 - phi) / phi)
        self.theta[1:] = -sigma_inv.dot(mu_0 - mu_1)

        if self.verbose:
            print('Final theta (GDA): {}'.format(self.theta))

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """

        y_hat = self._sigmoid(x.dot(self.theta))

        return y_hat

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
