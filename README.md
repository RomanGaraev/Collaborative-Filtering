# Collaborative-Filtering
Implementation of recommendation system for movies. Data set - is a short version of MovieLens-20M with 1M first ratings only.

The project contains two different recommenders. 

First, named **Basic model**, is a factorization of rating matrix into user matrix **P** and item matrix **Q**. 
Training of these matricies are made by `numpy`. Also, `scipy coo_matrix` was used for optimization.

The second recommender, **DL model**,  is a simple neural network. The only interisting feature of it is an input. **DL model** transforms ID of user and ID of movie to 
inner vectors representations, concatinate them together, and feed it to the hidden layers.

## Repository Structure

```
├── requirements.txt         <- The requirements file for reproducing the experiments. Most of them are standart and maybe already installed
├── Dockerfile               <- Commands to Docker, which will create the whole environment and run test.py 
|
├── data                     <- Data files directory
│   └── data                 <- Tables, where each row is (userId, movieId, rating)
│       └── test.csv         
│       └── train.csv     
|
├── models                   <- Saved pretrained models
│   ├── P.npy                <- Matrices from Basic model
│   └── Q.np                 <- 
│   └── recommender.pt       <- DL model
|
├── BasicModel.py            <- Matrix factorization method
├── DLModel.py               <- Pytorch deep model
├── train.py                 <- Invoke of training functions from BasicModel.Py and DLModel.py
├── test.py                  <- Predictions for some user 
```
## Dependences installing 
All the libraries can be pip installed using `pip install -r requirements.txt`

Python >=3.6 is required

## Running
Train and test scripts are simply running by `python train.py` and `python test.py` respectively.
