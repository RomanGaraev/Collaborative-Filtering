from os.path import join
from torch import nn
import torch


def load(frame):
    users = frame['userId'].values - 1
    movies = frame['movieId'].values - 1
    rates = frame['rating'].values
    n_samples = len(rates)
    batch_sz = 128
    n_users, n_movies = max(users) + 1, max(movies) + 1
    batches = []
    # Create batches
    for i in range(0, n_samples, batch_sz):
        limit = min(i + batch_sz, n_samples)
        batches.append((torch.tensor(users[i: limit], dtype=torch.long),
                        torch.tensor(movies[i: limit], dtype=torch.long),
                        torch.tensor(rates[i: limit], dtype=torch.float)))
    return batches, n_users, n_movies


class RecommenderNet(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=25, embedding_dropout=0.02, dropout_rate=0.2):
        super().__init__()
        self.n_users = n_users
        self.n_movies = n_movies
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden = nn.Sequential(nn.Linear(2*n_factors, 128),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(128, 128),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU())
        self.fc = nn.Linear(128, 1)
        self._init()
        self.to(self.device)

    def forward(self, users, movies):
        features = torch.cat([self.u(users), self.m(movies)], dim=1)
        x = self.drop(features)
        x = self.hidden(x)
        out = torch.sigmoid(self.fc(x))
        min_rating, max_rating = 1, 5
        out = out * (max_rating - min_rating) + min_rating
        return out

    def _init(self):
        """
        Initialize embeddings and hidden layers weights with xavier.
        """

        def init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.u.weight.data.uniform_(-0.05, 0.05)
        self.m.weight.data.uniform_(-0.05, 0.05)
        self.hidden.apply(init)
        init(self.fc)

    def Train(self, data, epochs=30):
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6,
                                                               factor=0.3, patience=2, verbose=True)
        errors = []
        for epoch in range(epochs):
            train_loss = 0
            for users_batch, movies_batch, rates_batch in data:
                self.zero_grad()
                out = self.forward(users_batch.to(self.device), movies_batch.to(self.device)).squeeze()
                loss = criterion(rates_batch.to(self.device), out)
                loss.backward()
                optimizer.step()
                train_loss += loss
            scheduler.step(loss)
            err = train_loss / len(data)
            errors.append(err)
            print(f"Loss {err} at epoch {epoch}")
        return errors

    def Test(self, data):
        self.eval()
        train_loss = 0
        criterion = nn.MSELoss(reduction='mean')
        for users_batch, movies_batch, rates_batch in data:
            out = self.forward(users_batch.to(self.device), movies_batch.to(self.device)).squeeze()
            loss = criterion(rates_batch.to(self.device), out)
            train_loss += loss
        print("Test MSE ", train_loss / len(data))

    def save(self):
        torch.save(self.state_dict(), join("models", "recommender.pt"))

    def load(self):
        self.load_state_dict(torch.load(join("models", "recommender.pt"), map_location=self.device))

    def predict(self, index, k=10):
        pred = [self.forward(torch.tensor(data=[index]), torch.tensor(data=[x])).squeeze().item()
                for x in range(self.n_movies)]
        pred = list(zip(range(self.n_movies), pred))
        return sorted(pred, key=lambda x: x[1], reverse=True)
