# Naive-Bayes
Simple Naive-Bayes program written in Python 

[Documentation]

  Defining File Paths:

  These variables hold the paths to the training and test data files.

    test_file = '/Users/pjotr/PycharmProjects/Naive-Bayes/Naive-Bayes-Program/res/agaricus-lepiota.test.data'
    train_file = '/Users/pjotr/PycharmProjects/Naive-Bayes/Naive-Bayes-Program/res/agaricus-lepiota.data'

  Load Data Function:

  This function reads data from a CSV file using pandas. It assumes that the first column contains labels and the remaining columns contain features.

     def load_data(file):
        reader = pd.read_csv(file, header=None)
        x = reader.iloc[:, 1:]  # Features
        y = reader.iloc[:, 0]   # Labels
        return x, y

  Step by step:

pd.read_csv(file, header=None): Reads the CSV file specified by file using pandas read_csv function, assuming no header.

reader.iloc[:, 1:]: Selects all rows and all columns starting from the second column, representing the features.

reader.iloc[:, 0]: Selects all rows and only the first column, representing the labels.

return x, y: Returns the features (x) and labels (y).

  Calculate Prior Function:

  This function calculates the prior probabilities of each class based on the frequency of occurrence in the training data.

    def calculate_prior(train_y):
        label_count = train_y.value_counts()  
        return label_count / len(train_y)       

Step by step: 

train_y.value_counts(): Counts the occurrences of each unique value in the Series train_y, effectively counting the occurrences of each label.

return label_count / len(train_y): Divides the count of each label by the total number of samples (len(train_y)), obtaining the prior probabilities.

  Laplace Smoothing Function:

  This function applies Laplace smoothing to the conditional probabilities of each feature given the class.

    def laplace_smoothing(label_data, train_x):
        cond_probabilities = {}
        for col in train_x.columns:  # Iterate over features
            value_counts = label_data[col].value_counts(normalize=True).to_dict() 
            total_values = len(label_data) + len(train_x[col].unique())
            for value in train_attributes[col].unique():
              count = value_counts.get(value, 0)
              smoothed_count = count + 1
              probability = smoothed_count / total_values
              cond_probabilities[col][value] = probability
          return cond_probabilities

Step by step:

label_data[col].value_counts(normalize=True).to_dict(): Counts the occurrences of each unique value in the column col of label_data and converts it to a dictionary with normalized probabilities.

total_values = len(label_data) + len(train_x[col].unique()): Calculates the total number of unique values in the column col, including missing values, and adds it to the length of label_data. This is used for Laplace smoothing.

count = value_counts.get(value, 0): Get the count of occurrences of the current value in the label_data dictionary. If the value is missing, default to 0.

smoothed_count = count + 1: Apply Laplace smoothing by adding 1 to the count.

probability = smoothed_count / total_values: Calculate the conditional probability for the current value in the current column using Laplace smoothing.

cond_probabilities[col][value] = probability: Update the cond_probabilities dictionary with the calculated probability for the current value in the current column.

  Fit Model Function:

  This function fits the Naive Bayes model by calculating prior probabilities and conditional probabilities for each feature.

    def fit_model(train_x, train_y):
        unique_labels = train_y.unique() # Get unique labels in training data
        prior_probabilities = calculate_prior(train_y)
        cond_probabilities = {}
        for label in unique_labels: # Iterate over unique labels
            label_data = train_x[train_y == label] # Filter data for current label
            cond_probabilities[label] = laplace_smoothing(label_data, train_x) 
        return prior_probabilities, cond_probabilities

Step by step:

train_y.unique(): Retrieves unique labels from train_y.

label_data = train_x[train_y == label]: Filters the features (train_x) based on the current label (label).

cond_probabilities[label] = laplace_smoothing(label_data, train_x): Calculates conditional probabilities for each feature given the label using Laplace smoothing.

  Prediction Functions:

  This function predicts the label for a single instance using the Naive Bayes classifier.

    def predict_instance(row, prior, cond_probabilities):
        max_prob = -1
        max_label = None
        for label, prior_prob in prior.items(): 
            prob = prior_prob
            for col, value in row.items(): 
                if value in cond_probabilities[label][col]:
                    prob *= cond_probabilities[label][col][value] 
        if prob > max_prob: 
            max_prob = prob
            max_label = label
    return max_label

Step by step:

for label, prior_prob in prior.items(): Iterates over each class (label) and its prior probability (prior_prob).

for col, value in row.items(): Iterates over each feature (col) and its value (value) in the current instance (row).

prob *= cond_probabilities[label][col][value]: Updates the probability by multiplying it with the corresponding conditional probability.

if prob > max_prob: max_prob = prob max_label = label: If the calculated probability is greater than the current maximum probability max_prob, update max_prob and max_label with the new values.

  Accuracy Function:

  These functions calculate accuracy of given model.

    def accuracy(true_labels, predicted_labels):
    correct = 0
    total = len(true_labels)
    for i in range(total):
        if true_labels[i] == predicted_labels[i]:
            correct += 1
    return correct / total

  Evaluate Function:

  This function computes all evaluation metrics at once.
  
    def evaluate(y_true, y_pred):
        acc = accuracy(y_true, y_pred)
        prec = precision(y_true, y_pred)
        rec = recall(y_true, y_pred)
        f = f_measure(y_true, y_pred)
        return acc, prec, rec, f

Step by step:

Calls the individual evaluation metric functions and returns the calculated values.

  Program Interface:
  
The rest of the code is for the program interface. It allows the user to train the model, test the model, print model info, or exit the program based on user input.
