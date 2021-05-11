"""
Invoke of DLModel and BasicModel methods. Result will be saved
"""
from DLModel import load, RecommenderNet
import matplotlib.pyplot as plt
from BasicModel import Sparse
from os.path import join
from numpy import log
import pandas as pd


def plot(errors, title):
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("MSE(log)")
    plt.grid()
    plt.plot(log(errors))
    plt.show()


if __name__ == "__main__":
    train_frame = pd.read_csv(join("data", "train.csv"))
    test_frame = pd.read_csv(join("data", "test.csv"))

    print("Basic sparse model is training...\n")
    basic_model = Sparse()
    basic_errors = basic_model.Train(train_frame)
    basic_model.Test(test_frame)
    basic_model.save()
    plot(basic_errors, title="Matrix factorization training loss")

    print("Deep model is training...\n")
    train_batches, n_users, n_movies = load(train_frame)
    net = RecommenderNet(n_users, n_movies)
    print(net)
    net_errors = net.Train(train_batches)
    test_batches, _, _ = load(test_frame)
    net.Test(test_batches)
    net.save()
    plot(net_errors, title="Network training loss")
