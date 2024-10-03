#SVM model
# SVM Code
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
from skimage.color import rgb2gray
import os

bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_Bleeding"
non_bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_NonBleeding"

# Loading and preprocessing the images-grayscale conversion of images
def load_and_preprocess_images(folder_path):
    images = []
    for file in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, file))
        image_gray = rgb2gray(image)
        images.append(image_gray)
    return images

bleeding_images = load_and_preprocess_images(bleeding_folder)
non_bleeding_images = load_and_preprocess_images(non_bleeding_folder)

# Feature extraction-Extraction of the HOG feature
def calculate_hog_features(image):
    return hog(image, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2))

bleeding_features = np.array([calculate_hog_features(image) for image in bleeding_images])
non_bleeding_features = np.array([calculate_hog_features(image) for image in non_bleeding_images])

# Combining HOG features using vstack
X = np.vstack((bleeding_features, non_bleeding_features))

# Creating labels for bleeding and non-bleeding images
bleeding_labels = np.ones(len(bleeding_images))
non_bleeding_labels = np.zeros(len(non_bleeding_images))
y = np.concatenate((bleeding_labels, non_bleeding_labels))

# Splitting data into train and test data (80:20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the SVM model
model = SVC(kernel='poly', degree=3, C=1.0)
model.fit(X_train, y_train)

# Evaluating the SVM model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Confusion Matrix, Precision, Recall, F1 Score
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("True Negative (TN):", conf_matrix[0,0])
print("False Negative (FN):", conf_matrix[1,0])
print("False Positive (FP):", conf_matrix[0,1])
print("True Positive (TP):", conf_matrix[1,1])
print(conf_matrix)
print("\nEvaluation Metrics:")
acc=(conf_matrix[1, 1]+conf_matrix[0, 0]) / (conf_matrix[1, 1] + conf_matrix[0, 1]+conf_matrix[1, 0] + conf_matrix[0, 0])
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)
print("Accuracy",acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

#Logistic Regression Model
#Logistic Regression Code
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from skimage.color import rgb2gray
import os

bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_Bleeding"
non_bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_NonBleeding"

# Load and preprocess images
def load_and_preprocess_images(folder_path):
    images = []
    for file in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, file))
        # Convert the image to grayscale
        image_gray = rgb2gray(image)
        images.append(image_gray)
    return images

bleeding_images = load_and_preprocess_images(bleeding_folder)
non_bleeding_images = load_and_preprocess_images(non_bleeding_folder)

# Feature extraction (HOG)
def calculate_hog_features(image):
    return hog(image, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2))

bleeding_features = np.array([calculate_hog_features(image) for image in bleeding_images])
non_bleeding_features = np.array([calculate_hog_features(image) for image in non_bleeding_images])

# Combine HOG features using vstack
X = np.vstack((bleeding_features, non_bleeding_features))

# Creating labels for each type of image
bleeding_labels = np.ones(len(bleeding_images))
non_bleeding_labels = np.zeros(len(non_bleeding_images))
y = np.concatenate((bleeding_labels, non_bleeding_labels))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the Logistic Regression model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Confusion Matrix, Precision, Recall, F1 Score
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("True Negative (TN):", conf_matrix[0,0])
print("False Negative (FN):", conf_matrix[1,0])
print("False Positive (FP):", conf_matrix[0,1])
print("True Positive (TP):", conf_matrix[1,1])
print(conf_matrix)
print("\nEvaluation Metrics:")
acc=(conf_matrix[1, 1]+conf_matrix[0, 0]) / (conf_matrix[1, 1] + conf_matrix[0, 1]+conf_matrix[1, 0] + conf_matrix[0, 0])
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)
print("Accuracy",acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

#Decision Tree Model
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from skimage.color import rgb2gray
import os

# Specify the paths to your image folders
bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_Bleeding"
non_bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_NonBleeding"

# Load and preprocess images
def load_and_preprocess_images(folder_path):
    images = []
    for file in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, file))
        # Convert the image to grayscale
        image_gray = rgb2gray(image)
        images.append(image_gray)
    return images

bleeding_images = load_and_preprocess_images(bleeding_folder)
non_bleeding_images = load_and_preprocess_images(non_bleeding_folder)

# Feature extraction (HOG)
def calculate_hog_features(image):
    return hog(image, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2))

bleeding_features = np.array([calculate_hog_features(image) for image in bleeding_images])
non_bleeding_features = np.array([calculate_hog_features(image) for image in non_bleeding_images])

# Combine HOG features
X = np.vstack((bleeding_features, non_bleeding_features))

# Create labels
bleeding_labels = np.ones(len(bleeding_images))
non_bleeding_labels = np.zeros(len(non_bleeding_images))
y = np.concatenate((bleeding_labels, non_bleeding_labels))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the Decision Tree model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Confusion Matrix, Precision, Recall, F1 Score
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("True Negative (TN):", conf_matrix[0,0])
print("False Negative (FN):", conf_matrix[1,0])
print("False Positive (FP):", conf_matrix[0,1])
print("True Positive (TP):", conf_matrix[1,1])
print(conf_matrix)
print("\nEvaluation Metrics:")
acc=(conf_matrix[1, 1]+conf_matrix[0, 0]) / (conf_matrix[1, 1] + conf_matrix[0, 1]+conf_matrix[1, 0] + conf_matrix[0, 0])
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)
print("Accuracy",acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

#Ensembled Model
#Ensembled Model
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from skimage.color import rgb2gray
import os

# Specify the paths to your image folders
bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_Bleeding"
non_bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_NonBleeding"

# Load and preprocess images
def load_and_preprocess_images(folder_path):
    images = []
    for file in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, file))
        # Convert the image to grayscale
        image_gray = rgb2gray(image)
        images.append(image_gray)
    return images

bleeding_images = load_and_preprocess_images(bleeding_folder)
non_bleeding_images = load_and_preprocess_images(non_bleeding_folder)

# Feature extraction (HOG)
def calculate_hog_features(image):
    return hog(image, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2))

bleeding_features = np.array([calculate_hog_features(image) for image in bleeding_images])
non_bleeding_features = np.array([calculate_hog_features(image) for image in non_bleeding_images])

# Combine HOG features
X = np.vstack((bleeding_features, non_bleeding_features))

# Create labels
bleeding_labels = np.ones(len(bleeding_images))
non_bleeding_labels = np.zeros(len(non_bleeding_images))
y = np.concatenate((bleeding_labels, non_bleeding_labels))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
lr_model = LogisticRegression()
svm_model = SVC(kernel='polynomial',degree=3,c=1)
dt_model = DecisionTreeClassifier()

# Create the ensemble using VotingClassifier
ensemble_model = VotingClassifier(estimators=[('lr', lr_model), ('svm', svm_model), ('dt', dt_model)], voting='hard')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Evaluate the ensemble model
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Confusion Matrix, Precision, Recall, F1 Score
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("True Negative (TN):", conf_matrix[0,0])
print("False Negative (FN):", conf_matrix[1,0])
print("False Positive (FP):", conf_matrix[0,1])
print("True Positive (TP):", conf_matrix[1,1])
print(conf_matrix)
print("\nEvaluation Metrics:")
acc=(conf_matrix[1, 1]+conf_matrix[0, 0]) / (conf_matrix[1, 1] + conf_matrix[0, 1]+conf_matrix[1, 0] + conf_matrix[0, 0])
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)
print("Accuracy",acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:",)

#Additional Experiments
#Random Forest Classifier
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from skimage.color import rgb2gray
import os

# Specify the paths to your image folders
bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_Bleeding"
non_bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_NonBleeding"

# Load and preprocess images
def load_and_preprocess_images(folder_path):
    images = []
    for file in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, file))
        # Convert the image to grayscale
        image_gray = rgb2gray(image)
        images.append(image_gray)
    return images

bleeding_images = load_and_preprocess_images(bleeding_folder)
non_bleeding_images = load_and_preprocess_images(non_bleeding_folder)

# Feature extraction (HOG)
def calculate_hog_features(image):
    return hog(image, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2))

bleeding_features = np.array([calculate_hog_features(image) for image in bleeding_images])
non_bleeding_features = np.array([calculate_hog_features(image) for image in non_bleeding_images])

# Combine HOG features
X = np.vstack((bleeding_features, non_bleeding_features))

# Create labels
bleeding_labels = np.ones(len(bleeding_images))
non_bleeding_labels = np.zeros(len(non_bleeding_images))
y = np.concatenate((bleeding_labels, non_bleeding_labels))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the Random Forest model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Confusion Matrix, Precision, Recall, F1 Score
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("True Negative (TN):", conf_matrix[0,0])
print("False Negative (FN):", conf_matrix[1,0])
print("False Positive (FP):", conf_matrix[0,1])
print("True Positive (TP):", conf_matrix[1,1])
print(conf_matrix)
print("\nEvaluation Metrics:")
acc=(conf_matrix[1, 1]+conf_matrix[0, 0]) / (conf_matrix[1, 1] + conf_matrix[0, 1]+conf_matrix[1, 0] + conf_matrix[0, 0])
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)
print("Accuracy",acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

#XGBoost
import cv2
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from skimage.color import rgb2gray
import os

# Specify the paths to your image folders
bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_Bleeding"
non_bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_NonBleeding"

# Load and preprocess images
def load_and_preprocess_images(folder_path):
    images = []
    for file in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, file))
        # Convert the image to grayscale
        image_gray = rgb2gray(image)
        images.append(image_gray)
    return images

bleeding_images = load_and_preprocess_images(bleeding_folder)
non_bleeding_images = load_and_preprocess_images(non_bleeding_folder)

# Feature extraction (HOG)
def calculate_hog_features(image):
    return hog(image, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2))

bleeding_features = np.array([calculate_hog_features(image) for image in bleeding_images])
non_bleeding_features = np.array([calculate_hog_features(image) for image in non_bleeding_images])

# Combine HOG features
X = np.vstack((bleeding_features, non_bleeding_features))

# Create labels
bleeding_labels = np.ones(len(bleeding_images))
non_bleeding_labels = np.zeros(len(non_bleeding_images))
y = np.concatenate((bleeding_labels, non_bleeding_labels))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Evaluate the XGBoost model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Confusion Matrix, Precision, Recall, F1 Score
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("True Negative (TN):", conf_matrix[0,0])
print("False Negative (FN):", conf_matrix[1,0])
print("False Positive (FP):", conf_matrix[0,1])
print("True Positive (TP):", conf_matrix[1,1])
print(conf_matrix)
print("\nEvaluation Metrics:")
acc=(conf_matrix[1, 1]+conf_matrix[0, 0]) / (conf_matrix[1, 1] + conf_matrix[0, 1]+conf_matrix[1, 0] + conf_matrix[0, 0])
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)
print("Accuracy",acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

#Neural Networks
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

# Specify the paths to your image folders
bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_Bleeding"
non_bleeding_folder = "/content/drive/MyDrive/DatsetForUse/Images_NonBleeding"

# Load and preprocess images
def load_and_preprocess_images(folder_path):
    images = []
    labels = []
    for file in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path, file))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
            image = cv2.resize(image, (128, 128))  # Resize images to a common size
            images.append(image)
            labels.append(1 if "Bleeding" in folder_path else 0)  # Assign labels

    return images, labels

bleeding_images, bleeding_labels = load_and_preprocess_images(bleeding_folder)
non_bleeding_images, non_bleeding_labels = load_and_preprocess_images(non_bleeding_folder)

# Combine images and labels
X = np.array(bleeding_images + non_bleeding_images)
y = np.array(bleeding_labels + non_bleeding_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert lists to NumPy arrays
X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(y_train), np.array(y_test)

# Create a CNN model
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with validation data
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", test_accuracy)
