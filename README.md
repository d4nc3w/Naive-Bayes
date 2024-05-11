# Naive-Bayes
Simple Naive-Bayes program written in Python 

[Documentation]

  Defining File Paths:

    test_file = '/Users/pjotr/PycharmProjects/Naive-Bayes/Naive-Bayes-Program/res/agaricus-lepiota.test.data'
    train_file = '/Users/pjotr/PycharmProjects/Naive-Bayes/Naive-Bayes-Program/res/agaricus-lepiota.data'
  
    
These variables hold the paths to the training and test data files.

  Load Data Function:

     def load_data(file):
        reader = pd.read_csv(file, header=None)
        x = reader.iloc[:, 1:]  # Features
        y = reader.iloc[:, 0]   # Labels
        return x, y
    
This function reads data from a CSV file using pandas. It assumes that the first column contains labels and the remaining columns contain features.

  Calculate Prior Function:

    def calculate_prior(train_y):
        label_count = train_y.value_counts()  
        return label_count / len(train_y)       
    
This function calculates the prior probabilities of each class based on the frequency of occurrence in the training data.

  Laplace Smoothing Function:

    def laplace_smoothing(label_data, train_x):
        cond_probabilities = {}
        for col in train_x.columns:  # Iterate over features
            value_counts = label_data[col].value_counts(normalize=True).to_dict() # Count occurrences of each value
            total_values = len(label_data) + len(train_x[col].unique())  # Total number of values for Laplace smoothing
            ...
        
This function applies Laplace smoothing to the conditional probabilities of each feature given the class.

  Fit Model Function:

    def fit_model(train_x, train_y):
        unique_labels = train_y.unique() # Get unique labels in training data
        prior_probabilities = calculate_prior(train_y) # Calculate prior probabilities
        cond_probabilities = {}
        for label in unique_labels: # Iterate over unique labels
            label_data = train_x[train_y == label] # Filter data for current label
            cond_probabilities[label] = laplace_smoothing(label_data, train_x) # Apply Laplace smoothing
        return prior_probabilities, cond_probabilities
    
This function fits the Naive Bayes model by calculating prior probabilities and conditional probabilities for each feature.

  Prediction Functions:

    def predict_instance(row, prior, cond_probabilities):
        max_prob = -1
        max_label = None
        for label, prior_prob in prior.items(): # Iterate over classes
            prob = prior_prob # Initialize probability
            for col, value in row.items(): # Iterate over features
                if value in cond_probabilities[label][col]: # Check if value is in conditional probabilities
                    prob *= cond_probabilities[label][col][value] # Multiply probability by conditional probability
                    ...
                    
This function predicts the label for a single instance using the Naive Bayes classifier.

  Evaluation Metrics Functions:

    def accuracy(y_true, y_pred):
        ...
    
These functions calculate various evaluation metrics like accuracy, precision, recall, and F-measure.

  Evaluate Function:
  
    def evaluate(y_true, y_pred):
        acc = accuracy(y_true, y_pred)
        prec = precision(y_true, y_pred)
        rec = recall(y_true, y_pred)
        f = f_measure(y_true, y_pred)
        return acc, prec, rec, f
    
This function computes all evaluation metrics at once.

  Program Interface:
The rest of the code is for the program interface. It allows the user to train the model, test the model, print model info, or exit the program based on user input.
