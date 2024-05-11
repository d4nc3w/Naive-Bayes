import pandas as pd

agaricus_train = '/Users/pjotr/PycharmProjects/res/agaricus-lepiota.data'
agaricus_test = '/Users/pjotr/PycharmProjects/res/agaricus-lepiota.test'

def load_data(file):
    reader = pd.read_csv(file, header=None)
    x = reader.iloc[:, 1:]
    y = reader.iloc[:, 0]
    return x, y


