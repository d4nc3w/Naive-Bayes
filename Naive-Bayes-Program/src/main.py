import pandas as pd

train_file = '/Users/pjotr/PycharmProjects/Naive-Bayes/Naive-Bayes-Program/res/agaricus-lepiota.data'
test_file = '/Users/pjotr/PycharmProjects/Naive-Bayes/Naive-Bayes-Program/res/agaricus-lepiota.test.data'

def load_data(file):
    reader = pd.read_csv(file, header=None)
    attributes = reader.iloc[:, 1:]
    labels = reader.iloc[:, 0]
    return attributes, labels

def calculate_prior(train_labels):
    # Counts each attribute
    label_count = train_labels.value_counts()
    return label_count / len(train_labels)

def calculate_cond_probabilities(label_data, train_attributes):
    cond_probabilities = {}
    # Iterate over attributes
    for col in train_attributes.columns:
        value_counts = label_data[col].value_counts(normalize=True).to_dict()
        total_values = len(label_data) + len(train_attributes[col].unique())

        cond_probabilities[col] = {}
        # Iterate over unique attributes
        for value in train_attributes[col].unique():
            # Get count of value or 0 if missing
            count = value_counts.get(value, 0)
            # Smoothing
            count += 1
            probability = count / total_values
            cond_probabilities[col][value] = probability
    return cond_probabilities

def fit_model(train_attributes, train_labels):
    unique_labels = train_labels.unique()
    prior_probabilities = calculate_prior(train_labels)
    cond_probabilities = {}
    # Iterate over unique labels
    for label in unique_labels:
        label_data = train_attributes[train_labels == label]
        # Calculate cond. probabilities
        cond_probabilities[label] = calculate_cond_probabilities(label_data, train_attributes)
    return prior_probabilities, cond_probabilities

def predict_instance(row, prior_probabilities, cond_probabilities):
    max_prob = 0
    prediction_label = None
    for label, prior_prob in prior_probabilities.items():
        # Initialize prior probability
        prob = prior_prob
        # Iterate over features
        for col, value in row.items():
            # Check if value is in conditional probabilities
            if value in cond_probabilities[label][col]:
                # Multiply probability by cond. prob.
                prob *= cond_probabilities[label][col][value]
        if prob > max_prob:
            max_prob = prob
            prediction_label = label
    return prediction_label

def predict(test_attributes, test_labels, prior_probabilities, cond_probabilities):
    predictions = []
    for i, row in test_attributes.iterrows():
        # Predict label for current row
        prediction_label = predict_instance(row, prior_probabilities, cond_probabilities)
        predictions.append(prediction_label)
        print("Predicted:", prediction_label, "Real:", test_labels.iloc[i])
    return predictions

def accuracy(true_labels, predicted_labels):
    correct = 0
    total = len(true_labels)
    for i in range(total):
        if true_labels[i] == predicted_labels[i]:
            correct += 1
    return correct / total

def precision(true_labels, predicted_labels):
    # True positives
    correct_p = sum(1 for i in range(len(true_labels)) if true_labels[i] == 'p' and predicted_labels[i] == 'p')
    # False positives
    wrong_p = sum(1 for i in range(len(true_labels)) if true_labels[i] == 'e' and predicted_labels[i] == 'p')
    return correct_p / (correct_p + wrong_p)

def recall(true_labels, predicted_labels):
    # True positives
    correct_p = sum(1 for i in range(len(true_labels)) if true_labels[i] == 'p' and predicted_labels[i] == 'p')
    # False negatives
    wrong_n = sum(1 for i in range(len(true_labels)) if true_labels[i] == 'p' and predicted_labels[i] == 'e')
    return correct_p / (correct_p + wrong_n)

def f_measure(true_labels, predicted_labels):
    p = precision(true_labels, predicted_labels)
    r = recall(true_labels, predicted_labels)
    return 2 * p * r / (p + r)

def get_model_info(true_labels, predicted_labels):
    acc = accuracy(true_labels, predicted_labels)
    prec = precision(true_labels, predicted_labels)
    rec = recall(true_labels, predicted_labels)
    f = f_measure(true_labels, predicted_labels)
    return acc, prec, rec, f

# --------------------------------------------
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
            acc, prec, rec, f_meas = get_model_info(test_labels, predictions)

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
            acc, prec, rec, f_meas = get_model_info(test_labels, predictions)
            print("----------Model Info----------")
            print("Accuracy:", acc)
            print("Precision:", prec)
            print("Recall:", rec)
            print("F-measure:", f_meas)
    if choice == 4:
        print("Closing...")
        print("------------------------------")
        exit()