import pandas as pd

train_file = '/Users/pjotr/PycharmProjects/Naive-Bayes/Naive-Bayes-Program/res/agaricus-lepiota.data'
test_file = '/Users/pjotr/PycharmProjects/Naive-Bayes/Naive-Bayes-Program/res/agaricus-lepiota.test.data'

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

    # Calculate cond. probabilities (Laplace)
    cond_probabilities = {}
    for label in unique_labels:
        label_data = train_x[train_y == label]
        cond_probabilities[label] = {}
        for col in train_x.columns:
            value_counts = label_data[col].value_counts(normalize=True).to_dict()
            total_values = len(label_data) + len(train_x[col].unique())  # Add Laplace smoothing
            for value in train_x[col].unique():
                if value not in value_counts:
                    value_counts[value] = 0
            cond_probabilities[label][col] = {value: (value_counts[value] + 1) / total_values for value in train_x[col].unique()}

    return prior_probabilities, cond_probabilities

def predict(x_test, prior, cond_probabilities):
    predictions = []
    for i, (_, row) in enumerate(x_test.iterrows()):
        max_prob = -1
        max_label = None
        for label, prior_prob in prior.items():
            prob = prior_prob
            for col, value in row.items():
                if value in cond_probabilities[label][col]:
                    prob *= cond_probabilities[label][col][value]
                else:
                    # Smoothing
                    prob *= 1 / (len(cond_probabilities[label][col]) + len(x_test[col].unique()))
            if prob > max_prob:
                max_prob = prob
                max_label = label
        predictions.append(max_label)
        print(f"Instance {i + 1}: Prediction - {max_label}, True Label - {test_y.iloc[i]}")
    return predictions

def accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def precision(y_true, y_pred):
    tp = sum((true == 'p' and pred == 'p') for true, pred in zip(y_true, y_pred))
    fp = sum((true == 'e' and pred == 'p') for true, pred in zip(y_true, y_pred))
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = sum((true == 'p' and pred == 'p') for true, pred in zip(y_true, y_pred))
    fn = sum((true == 'p' and pred == 'e') for true, pred in zip(y_true, y_pred))
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def f_measure(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)

def evaluate(y_true, y_pred):
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f = f_measure(y_true, y_pred)
    return acc, prec, rec, f

# Main
train_x, train_y = load_data(train_file)
test_x, test_y = load_data(test_file)

prior, cond_probabilities = fit_model(train_x, train_y)

predictions = predict(test_x, prior, cond_probabilities)

acc, prec, rec, f_meas = evaluate(test_y, predictions)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F-measure:", f_meas)
