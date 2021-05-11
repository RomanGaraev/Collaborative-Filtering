from numpy import random, multiply, sum, save, load, clip
from sklearn.metrics import mean_squared_error
from scipy.sparse.coo import coo_matrix
from os.path import join

'''
def get_matrix(frame):
    """
    Parse DataFrame from .csv file to get UxI matrix, where
    U - amount of users, I - amount of films,
    values in the cells are corresponding ratings
    """
    u_max = max(frame['userId'])
    i_max = max(frame['movieId'])
    ratings = np.zeros(shape=(u_max, i_max))
    for _, row in frame.iterrows():
        ratings[int(row['userId']) - 1][int(row['movieId']) - 1] = row['rating']
    return ratings


def vectorized_gradient(R, k=4, alpha=0.01, iterations=50, w=0.0001) -> (np.array, np.array):
    """
    :param R: matrix of ratings, output of get_matrix()
    :param k: dimensionality of representation matrices, P and Q
    :param alpha: learning rate
    :param iterations: amount of gradient steps
    :param w: weight decay
    :return: P, Q
    """
    P = np.random.rand(R.shape[0], k)
    Q = np.random.rand(R.shape[1], k)
    # All non-zero elements to 1
    M = np.clip(R, 0, 1)
    viewed = sum(sum(M))
    print(viewed, P.shape, Q.shape)
    for i in range(iterations):
        diff = np.multiply((P @ Q.transpose() - R), M)
        P_new = P - alpha * (w * P + diff @ Q)
        Q_new = Q - alpha * (w * Q + diff.T @ P)
        P = P_new
        Q = Q_new
        print(f"Iteration {i}, loss {sum(diff) / viewed}")
'''


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

    def predict(self, index, k=10):
        predictions = list(zip(range(len(self.Q)), clip((self.Q @ self.P[index].T), a_min=1, a_max=5)))
        return sorted(predictions, key=lambda x: x[1], reverse=True)