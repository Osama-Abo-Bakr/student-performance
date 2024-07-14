# ML - Student Performance

## Project Overview

This project aims to predict student performance using deep learning techniques. The process includes data analysis, preprocessing, model training, and evaluation to achieve high prediction accuracy.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries and Tools](#libraries-and-tools)
3. [Data Analysis and Visualization](#data-analysis-and-visualization)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
6. [Results](#results)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Conclusion](#conclusion)
10. [Contact](#contact)

## Introduction

This project leverages deep learning to predict student performance based on various features. By analyzing the given dataset, we apply a neural network model to predict the average scores of students.

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn**: Preprocessing and evaluation metrics
- **TensorFlow, Keras**: Deep learning model development

## Data Analysis and Visualization

1. **Data Loading**:
   - Loaded the dataset using `pd.read_csv()`.

2. **Data Summary**:
   - Checked for missing values and summarized the data using `data.info()` and `data.shape()`.

3. **Data Distribution**:
   - Visualized the distribution of scores using `sns.distplot()` and `sns.pairplot()`.

4. **Correlation Analysis**:
   - Used a heatmap to visualize correlations between features.

5. **Box Plots**:
   - Analyzed score distributions across different categories using `sns.boxplot()`.

## Data Preprocessing

1. **Label Encoding**:
   - Transformed categorical variables using `LabelEncoder`.

2. **Feature Engineering**:
   - Created a new feature `Average_score` by averaging math, reading, and writing scores.

3. **Data Splitting**:
   - Split the data into training and testing sets using `train_test_split()`.

## Modeling

1. **Model Architecture**:
   - Built a neural network model with multiple layers using Keras:
     - Input layer: `Dense(256, activation="relu")`
     - Hidden layers: `Dense(128, activation="relu")`, `Dense(64, activation="relu")`, `Dense(16, activation="relu")`
     - Output layer: `Dense(1, activation="linear")`

2. **Model Compilation**:
   - Compiled the model with the Adam optimizer and MSE loss function.

3. **Model Training**:
   - Trained the model for 500 epochs and validated on the test set.

## Results

- **Loss Curves**:
  - Plotted training and validation loss curves to monitor model performance.

- **Evaluation Metrics**:
  - Mean Squared Error (MSE): 0.00502
  - R² Score: 0.999977

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/student-performance.git
   ```

2. Navigate to the project directory:
   ```bash
   cd student-performance
   ```


## Usage

1. **Prepare Data**:
   - Ensure the dataset is available at the specified path.

2. **Train Model**:
   - Run the provided script to train the neural network model and evaluate its performance.

3. **Predict Outcomes**:
   - Use the trained model to predict student performance on new data.

## Conclusion

This project demonstrates the use of deep learning techniques to predict student performance. The neural network model achieved high accuracy, providing valuable insights into the factors influencing student scores.

## Contact

For questions or collaborations, please reach out via:

- **Email**: [Gmail](mailto:osamaoabobakr12@gmail.com)
- **LinkedIn**: [LinkedIn](https://linkedin.com/in/osama-abo-bakr-293614259/)

---

### Sample Code (for reference)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import tensorflow.keras as k

# Reading Data
data = pd.read_csv(r"D:\Courses language programming\6.1_Deep Learning (Udemy)\PDF\CSV & IPNB\8 - StudentsPerformance(Project_2).csv")
data.head(5)

# Data Analysis
data.isnull().sum()
data.info()
data.shape

# Visualization
sns.distplot(data["math score"])
plt.title("Distribution For Math Score")

df = ["math score", "reading score", "writing score"]
sns.pairplot(data[df])
sns.heatmap(data[df].corr(), annot=True, square=True)

sns.boxenplot(x="gender", y="math score", data=data)
sns.boxplot(x="gender", y="math score", data=data)
plt.figure(figsize=(10, 10))
sns.boxplot(x="parental level of education", y="math score", data=data)
plt.figure(figsize=(10, 10))
sns.boxplot(x="parental level of education", y="reading score", data=data)
plt.figure(figsize=(10, 10))
sns.boxplot(x="test preparation course", y="math score", data=data)

# Label Encoding
la = LabelEncoder()
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = la.fit_transform(data[col])

# Feature Engineering
data["Average_score"] = (data["math score"] + data["writing score"] + data["reading score"]) / 3
data.head(5)

# Splitting Data
X = data.drop(columns="Average_score", axis=1)
Y = data["Average_score"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

# Building Model
model_tf = k.models.Sequential()
model_tf.add(k.layers.Dense(256, activation="relu"))
model_tf.add(k.layers.Dense(128, activation="relu"))
model_tf.add(k.layers.Dense(64, activation="relu"))
model_tf.add(k.layers.Dense(16, activation="relu"))
model_tf.add(k.layers.Dense(1, activation="linear"))

model_tf.compile(optimizer="Adam", loss="mse")
history = model_tf.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))

# Plotting Loss Curves
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("EPOCHS")
plt.ylabel("Loss")
plt.legend()
plt.grid()

# Model Evaluation
prediction = model_tf.predict(x_test)
print(f"The Mean Squared Error is ==> {mean_squared_error(prediction, y_test)}")
print(f"The R² Score is ==> {r2_score(prediction, y_test)}")
```
