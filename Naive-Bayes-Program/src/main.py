import pandas as pd

train_file = '/Users/pjotr/PycharmProjects/Naive-Bayes/Naive-Bayes-Program/res/agaricus-lepiota.data'
test_file = '/Users/pjotr/PycharmProjects/Naive-Bayes/Naive-Bayes-Program/res/agaricus-lepiota.test.data'

def load_data(file):
    reader = pd.read_csv(file, header=None)
    x = reader.iloc[:, 1:]  # Features
    y = reader.iloc[:, 0]   # Labels
    return x, y

def calculate_prior(train_y):
    label_count = train_y.value_counts()    # Count occurrences of each label
    return label_count / len(train_y)       # Calculate probabilities

def laplace_smoothing(label_data, train_x):
    cond_probabilities = {}
    for col in train_x.columns:  # Iterate over features
        value_counts = label_data[col].value_counts(normalize=True).to_dict() # Count occurrences of each value
        total_values = len(label_data) + len(train_x[col].unique())  # Total number of values for Laplace smoothing

        cond_probabilities[col] = {}
        for value in train_x[col].unique(): # Iterate over unique values in feature
            count = value_counts.get(value, 0) # Get count of value
            smoothed_count = count + 1 # Laplace
            probability = smoothed_count / total_values
            cond_probabilities[col][value] = probability

    return cond_probabilities

def fit_model(train_x, train_y):
    unique_labels = train_y.unique() # Get unique labels in training data
    prior_probabilities = calculate_prior(train_y) # Calculate prior probabilities
    cond_probabilities = {}
    for label in unique_labels: # Iterate over unique labels
        label_data = train_x[train_y == label] # Filter data for current label
        cond_probabilities[label] = laplace_smoothing(label_data, train_x) # Apply Laplace smoothing
    return prior_probabilities, cond_probabilities

def predict_instance(row, prior, cond_probabilities):
    max_prob = -1
    max_label = None
    for label, prior_prob in prior.items(): # Iterate over classes
        prob = prior_prob # Initialize probability
        for col, value in row.items(): # Iterate over features
            if value in cond_probabilities[label][col]: # Check if value is in conditional probabilities
                prob *= cond_probabilities[label][col][value] # Multiply probability by conditional probability
        if prob > max_prob: # Update maximum probability and label if current probability is higher
            max_prob = prob
            max_label = label
    return max_label

def predict(x_test, prior, cond_probabilities):
    predictions = []
    for _, row in x_test.iterrows(): # Iterate over test instances
        max_label = predict_instance(row, prior, cond_probabilities) # Predict label for current instance
        predictions.append(max_label) # Append predicted label to list
    return predictions

def accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) # Count correct predictions
    return correct / len(y_true)

def precision(y_true, y_pred):
    correct_p = sum((true == 'p' and pred == 'p') for true, pred in zip(y_true, y_pred)) # Count true positives
    wrong_p = sum((true == 'e' and pred == 'p') for true, pred in zip(y_true, y_pred)) # Count false positives
    if correct_p + wrong_p == 0:
        return 0
    return correct_p / (correct_p + wrong_p)

def recall(y_true, y_pred):
    correct_p = sum((true == 'p' and pred == 'p') for true, pred in zip(y_true, y_pred))  # Count true positives
    wrong_n = sum((true == 'p' and pred == 'e') for true, pred in zip(y_true, y_pred)) # Count false negatives
    if correct_p + wrong_n == 0:
        return 0
    return correct_p / (correct_p + wrong_n)

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

# Program Interface
isTrained = False
isTested = False
while True:
    print("-------------MENU-------------")
    print("(1) Train Model")
    print("(2) Test Model")
    print("(3) Print Model Info")
    print("(4) Exit")
    print("------------------------------")
    choice = int(input("Enter your choice: "))
    if choice == 1:
        print("Training Model...")
        train_x, train_y = load_data(train_file)
        prior, cond_probabilities = fit_model(train_x, train_y)
        isTrained = True
        print("Model trained successfully")
    if choice == 2:
        if (isTrained == False):
            print("Model not trained yet")
            continue
        else:
            print("Testing Model...")
            test_x, test_y = load_data(test_file)
            predictions = predict(test_x, prior, cond_probabilities)
            acc, prec, rec, f_meas = evaluate(test_y, predictions)
            print("----------Model Info----------")
            print("Accuracy:", acc)
            print("Precision:", prec)
            print("Recall:", rec)
            print("F-measure:", f_meas)
            isTested = True
    if choice == 3:
        if (isTrained == False):
            print("Model not trained yet")
            continue
        elif(isTested == False):
            print("Model not tested yet")
            continue
        else:
            acc, prec, rec, f_meas = evaluate(test_y, predictions)
            print("----------Model Info----------")
            print("Accuracy:", acc)
            print("Precision:", prec)
            print("Recall:", rec)
            print("F-measure:", f_meas)
    if choice == 4:
        print("Closing...")
        print("------------------------------")
        exit()


