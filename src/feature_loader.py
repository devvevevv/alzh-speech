import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

FEATURES_PATH = r"..\data\features.csv"

class SpeechFeatureDataset(Dataset):
    def __init__(self, features, label):
        self.X = torch.tensor(features.values, dtype = torch.float32)
        self.y = torch.tensor(label.values, dtype = torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders():

    """
    Load the preprocessed features dataset, split it into training and testing sets,
    and return PyTorch DataLoaders for each.

    :return: pytorch DataLoaders
    """

    df = pd.read_csv(FEATURES_PATH)

    X = df.drop(columns = ["Label"])
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

    train_labels_array = y_train.values

    train_set = SpeechFeatureDataset(X_train, y_train)
    test_set = SpeechFeatureDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size = 32, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = 32, shuffle = False)

    return train_loader, test_loader, train_labels_array