"""
Matrix factorization of rating. Create P and Q - representation of users and items respectively
"""
from numpy import random, multiply, sum, save, load, clip
from sklearn.metrics import mean_squared_error
from scipy.sparse.coo import coo_matrix
from os.path import join


class Sparse:
    def Train(self, frame, k=5, alpha=0.0001, iterations=100, w=0.0001):
        """
            :param frame: pandas Dataframe with ratings
            :param k: dimensionality of representation matrices, P and Q
            :param alpha: learning rate
            :param iterations: amount of gradient steps
            :param w: weight decay
            :return: representation of users P and representation of items Q
            """
        # Parse data
        ratings = frame['rating'].values
        userIds = frame['userId'].values
        itemIds = frame['movieId'].values
        R = coo_matrix((ratings, (userIds, itemIds)))
        self.P = random.rand(R.shape[0], k)
        self.Q = random.rand(R.shape[1], k)
        errors = []
        for i in range(iterations):
            P_tau = self.P[userIds, :]
            Q_tau = self.Q[itemIds, :]

            # Inner product over second axis
            rating_pred = sum(multiply(P_tau, Q_tau), axis=1)
            # = (P @ Q - R), difference between prediction and real rating
            R_diff = coo_matrix((rating_pred, (userIds, itemIds))) - R

            # Update matrices
            P_new = self.P - alpha * (w * self.P + R_diff @ self.Q)
            Q_new = self.Q - alpha * (w * self.Q + R_diff.T @ self.P)
            self.P = P_new
            self.Q = Q_new
            err = mean_squared_error(ratings, rating_pred)
            errors.append(err)
            print(f"Iteration {i + 1}/{iterations}, MSE {err:.4f}")
        return errors

    def Test(self, frame):
        # The same procedure as during the training -
        # multiply P and Q, get predictions R^hat,
        # find the difference between it and real ratings
        ratings = frame['rating'].values
        userIds = frame['userId'].values
        itemIds = frame['movieId'].values
        P_tau = self.P[userIds, :]
        Q_tau = self.Q[itemIds, :]
        rating_pred = sum(multiply(P_tau, Q_tau), axis=1)
        print("Test MSE ", mean_squared_error(ratings, rating_pred))

    def save(self):
        save(join("models", "P.npy"), self.P)
        save(join("models", "Q.npy"), self.Q)

    def load(self):
        self.P = load(join("models", "P.npy"))
        self.Q = load(join("models", "Q.npy"))
        return self.P, self.Q

    def predict(self, index):
        """
        Prediction of ratings of each movie for some user with index
        :param index: "name" of user
        :return: list of pairs (#of movie, predicted rating)
        """
        predictions = list(zip(range(len(self.Q)), clip((self.Q @ self.P[index].T), a_min=1, a_max=5)))
        return sorted(predictions, key=lambda x: x[1], reverse=True)
    