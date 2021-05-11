"""
Get prediction for some particular user.
Top-10 prediction by the both algorithms and difference between real and predicted ratings are given
"""
from sklearn.metrics import mean_squared_error as MSE
from DLModel import load, RecommenderNet
from BasicModel import Sparse
from os.path import join
import pandas as pd


if __name__ == "__main__":
    index = 1
    print(f"Top 10 predictions of basic model for {index} user: (movieId, rating)")
    basic_model = Sparse()
    P, Q = basic_model.load()
    basic_pred = basic_model.predict(index)
    print(basic_pred[:10])

    print(f"Top 10 predictions of deep model for {index} user: (movieId, rating)")
    test_frame = pd.read_csv(join("data", "test.csv"))
    _, n_users, n_movies = load(test_frame)
    dl_model = RecommenderNet(n_users=n_users, n_movies=n_movies)
    dl_model.load()
    dl_pred = dl_model.predict(index)
    print(dl_pred[:10])

    # Create DataFrame with real ratings in test data set and predictions
    test_frame = test_frame[test_frame['userId'] == 1]
    id = test_frame['movieId'].values
    rating = test_frame['rating'].values
    basic_pred = [pair[1] for pair in basic_pred if pair[0] in id]
    dl_pred = [pair[1] for pair in dl_pred if pair[0] in id]
    print(pd.DataFrame(data={"Movie":id, "Real":rating, "Basic": basic_pred, "Deep": dl_pred}))
    print(f"MSE of basic method: {MSE(basic_pred, rating)}")
    print(f"MSE of DL method: {MSE(dl_pred, rating)}")