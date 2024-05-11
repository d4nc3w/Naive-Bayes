import pandas as pd

agaricus_train = '/Users/pjotr/PycharmProjects/res/agaricus-lepiota.data'
agaricus_test = '/Users/pjotr/PycharmProjects/res/agaricus-lepiota.test'

def load_data(file):
    reader = pd.read_csv(file, header=None)
    x = reader.iloc[:, 1:]
    y = reader.iloc[:, 0]
    return x, y

def fit_model(train_x, train_y):
    # Calculate prior
    unique_labels = train_y.unique()
    label_count = train_y.value_counts()
    num_of_instances = len(train_y)
    prior_probabilities = {label: label_count[label] / num_of_instances for label in unique_labels}

    # Calculate cond. probabilities
    cond_probabilities = {}
    for label in unique_labels:
        label_data = train_x[train_y == label]
        cond_probabilities[label] = {}
        for col in train_x.columns:
            value_counts = label_data[col].value_counts(normalize=True).to_dict()
            cond_probabilities[label][col] = value_counts

    return prior_probabilities, cond_probabilities




