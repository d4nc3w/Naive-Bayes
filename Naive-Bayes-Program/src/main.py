import pandas as pd

train_file = '/Users/pjotr/PycharmProjects/Naive-Bayes/Naive-Bayes-Program/res/agaricus-lepiota.data'
test_file = '/Users/pjotr/PycharmProjects/Naive-Bayes/Naive-Bayes-Program/res/agaricus-lepiota.test.data'

def load_data(file):
    reader = pd.read_csv(file, header=None)
    attributes = reader.iloc[:, 1:]
    labels = reader.iloc[:, 0]
    return attributes, labels

def calculate_prior(train_labels):
    label_count = train_labels.value_counts()    # Count occurrences of each label
    return label_count / len(train_labels)       # Calculate probabilities

def laplace_smoothing(label_data, train_attributes):
    cond_probabilities = {}
    for col in train_attributes.columns:  # Iterate over features
        value_counts = label_data[col].value_counts(normalize=True).to_dict() # Count occurrences of each value
        total_values = len(label_data) + len(train_attributes[col].unique())  # Total number of values for Laplace smoothing

        cond_probabilities[col] = {}
        for value in train_attributes[col].unique(): # Iterate over unique values in feature
            count = value_counts.get(value, 0) # Get count of value or 0 if missing
            smoothed_count = count + 1 # Laplace method
            probability = smoothed_count / total_values
            cond_probabilities[col][value] = probability

    return cond_probabilities

def fit_model(train_attributes, train_labels):
    unique_labels = train_labels.unique() # Get unique labels in training data
    prior_probabilities = calculate_prior(train_labels) # Calculate prior probabilities
    cond_probabilities = {}
    for label in unique_labels: # Iterate over unique labels
        label_data = train_attributes[train_labels == label] # Filter data for current label
        cond_probabilities[label] = laplace_smoothing(label_data, train_attributes) # Apply Laplace smoothing
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

def predict(test_attributes, test_labels, prior, cond_probabilities):
    predictions = []
    for i, row in test_attributes.iterrows(): # Iterate over test instances
        max_label = predict_instance(row, prior, cond_probabilities) # Predict label for current instance
        predictions.append(max_label) # Append predicted label to list
        print("Predicted:", max_label, "Actual:", test_labels.iloc[i])
    return predictions

def accuracy(true_labels, predicted_labels):
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred) # Count correct predictions
    return correct / len(true_labels)

def precision(true_labels, predicted_labels):
    correct_p = sum(1 for i in range(len(true_labels)) if true_labels[i] == 'p' and predicted_labels[i] == 'p')  # Count true positives
    wrong_p = sum(1 for i in range(len(true_labels)) if true_labels[i] == 'e' and predicted_labels[i] == 'p')  # Count false positives
    if correct_p + wrong_p == 0:
        return 0
    return correct_p / (correct_p + wrong_p)

def recall(true_labels, predicted_labels):
    correct_p = sum(1 for i in range(len(true_labels)) if true_labels[i] == 'p' and predicted_labels[i] == 'p')  # Count true positives
    wrong_n = sum(1 for i in range(len(true_labels)) if true_labels[i] == 'p' and predicted_labels[i] == 'e')  # Count false negatives
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
        train_attributes, train_labels = load_data(train_file)
        prior, cond_probabilities = fit_model(train_attributes, train_labels)
        isTrained = True
        print("Model trained successfully")
    if choice == 2:
        if (isTrained == False):
            print("Model not trained yet")
            continue
        else:
            print("Testing Model...")
            test_attributes, test_labels = load_data(test_file)
            predictions = predict(test_attributes, test_labels, prior, cond_probabilities)
            acc, prec, rec, f_meas = evaluate(test_labels, predictions)

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
            acc, prec, rec, f_meas = evaluate(test_labels, predictions)
            print("----------Model Info----------")
            print("Accuracy:", acc)
            print("Precision:", prec)
            print("Recall:", rec)
            print("F-measure:", f_meas)
    if choice == 4:
        print("Closing...")
        print("------------------------------")
        exit()