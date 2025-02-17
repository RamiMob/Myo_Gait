# Importing necessary libraries
import math  # Provides mathematical functions (e.g., sqrt, log, etc.)
import numpy as np  # Library for numerical computations and array manipulations
import pandas as pd  # Library for data manipulation and analysis (e.g., DataFrames)
import matplotlib.pyplot as plt  # Library for creating visualizations and plots
from matplotlib import patches
from sklearn.model_selection import train_test_split  # Splits data into training and testing sets
from sklearn.model_selection import KFold  # Implements k-fold cross-validation
from sklearn.preprocessing import StandardScaler  # Standardizes features by removing mean and scaling to unit variance
from libemg.filtering import Filter  # For filtering EMG signals (e.g., bandpass, notch filters)
from libemg.emg_predictor import EMGClassifier  # For building and training EMG classification models
from libemg.feature_extractor import FeatureExtractor  # Extracts features from EMG signals
from libemg.offline_metrics import OfflineMetrics  # Computes offline performance metrics for EMG models
from libemg.utils import get_windows  # Utility function to segment EMG data into windows

# Initialize an empty list to store accuracies
accuracies = []
fe = FeatureExtractor()
# Set the motion type and other parameters
mottype = 'WAK'
sf = 1920  # Sampling frequency
ws = 288   # Window size
si = 48    # Step interval
featsets = ["HTD", "TDPSD", "TDAR", "HJORTH"] # List of feature sets
models = ["LDA", "KNN", "SVM"] # List of classifiers
accdict = {}  # Dictionary to store accuracies for each classifier and feature set
fi = Filter(sf)
fi.install_filters(filter_dictionary={"name": "bandpass", "cutoff": [10, 400], "order": 4})
# Loop through each classifier
for classifier in models:
    # Set classifier-specific parameters
    if classifier == 'KNN':
        params = {'n_neighbors': 5}  # Set KNN parameters
    elif classifier == 'SVM':
        params = {'kernel': 'rbf', 'gamma': 'scale'}  # Set SVM parameters
    elif classifier == 'RF':
        params = {'n_estimators': 99, 'max_depth': 20, 'max_leaf_nodes': 10}  # Set RF parameters
    elif classifier == 'LDA':
        params = {}  # LDA has no specific parameters here

    # Loop through each feature set
    for feat in featsets:
        accuracies = []  # Reset accuracies list for each feature set

        # Loop through 40 subjects (1-40)
        for i in range(1, 41):
            # Determine file paths for data and labels based on subject number
            if i < 10:
                data_path = f'/content/drive/MyDrive/Colab Notebooks/Bohnes25/Sub0{i}_{mottype}_Data.csv'
                labels_path = f'/content/drive/MyDrive/Colab Notebooks/Bohnes25/Sub0{i}_{mottype}_Label.csv'
            else:
                data_path = f'/content/drive/MyDrive/Colab Notebooks/Bohnes25/Sub{i}_{mottype}_Data.csv'
                labels_path = f'/content/drive/MyDrive/Colab Notebooks/Bohnes25/Sub{i}_{mottype}_Label.csv'

            # Load the data and labels from CSV files
            data = pd.read_csv(data_path)
            labels = pd.read_csv(labels_path)
            # Extract sEMG columns and apply bandpass filter (10-400 Hz)
            emgcall = data.columns.str.contains('sEMG')
            semg = data.iloc[:, emgcall].to_numpy()
            # Get gait phases and trial numbers, removing NaN values
            gaitphase = labels.iloc[:, 1].to_numpy()
            trialno = labels.iloc[:, 2].to_numpy()
            nann = np.isnan(gaitphase)
            semg = semg[~nann]
            gaitphase = gaitphase[~nann] #Data cleaning
            # Initialize KFold cross-validation with 5 splits
            kf = KFold(n_splits=5, shuffle=False)

            # Loop through each fold of the cross-validation
            for i, (train_index, test_index) in enumerate(kf.split(semg)):
                datatrain, datatest = semg[train_index], semg[test_index]
                classtrain, classtest = gaitphase[train_index], gaitphase[test_index] # Split
                datatrain = fi.filter(datatrain)
                datatest = fi.filter(datatest) # Filter
                windowstrain=get_windows(datatrain, ws, si)
                windowstest=get_windows(datatest, ws, si) # Segment
                feattrain = fe.extract_feature_group(feat, windowstrain)
                feattest = fe.extract_feature_group(feat, windowstest) #Extract features
                feattrain_overall = np.hstack([value for value in feattrain.values()])
                feattest_overall = np.hstack([value for value in feattest.values()])
                classtrain = classtrain[::si][:feattrain_overall.shape[0]]
                classtest = classtest[::si][:feattest_overall.shape[0]] #Synchronizing labels
                sc = StandardScaler()
                feattrain_overall = sc.fit_transform(feattrain_overall)
                feattest_overall = sc.transform(feattest_overall) #Standardization
                data_set = {}
                data_set['training_features'] = feattrain_overall
                data_set['training_labels'] = classtrain
                model = EMGClassifier(classifier)
                model.fit(data_set, None, params) #Training the classifier
                # Add majority voting for predictions
                model.add_majority_vote(num_samples=3)
                # Get predictions and probabilities from the model
                preds, probs = model.run(feattest_overall)
                # Evaluate the classification accuracy using OfflineMetrics
                evm = OfflineMetrics()
                accuracies.append(evm.get_CA(classtest, preds + 1))  # Adjust labels by +1
        # Store the average accuracy for the classifier and feature set
        accdict[classifier, feat] = np.mean(accuracies)
# Print the results dictionary containing accuracies for each classifier and feature set
print(accdict)

# Extract unique models and feature sets
models = sorted(set(k[0] for k in accdict.keys()))
feature_sets = sorted(set(k[1] for k in accdict.keys()))

# Prepare data for plotting
x = np.arange(len(feature_sets))  # Feature set positions
width = 0.15  # Bar width
plt.figure()

# Create bars for each model
for i, model in enumerate(models):
    accuracies = [accdict.get((model, feature), 0) for feature in feature_sets]
    plt.bar(x + i * width, accuracies, width, label=model)

# Labeling
plt.xlabel("Feature Sets")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(x + width * (len(models) - 1) / 2, feature_sets)
plt.legend(title="Models")

# Adjust y-axis limits with padding
min_acc = min(accdict.values())
max_acc = max(accdict.values())
padding = (max_acc - min_acc) * 0.1
plt.ylim(min_acc - padding, max_acc + padding)

# Show plot without grid (default background)
plt.grid(False)
#plt.gca().set_facecolor('white')
plt.show()
