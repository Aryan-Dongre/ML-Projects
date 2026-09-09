# 🍷 Wine Quality Prediction

A Machine Learning project that analyzes physicochemical properties of red wine and uses a **Random Forest Classifier** to predict wine quality.

## 📌 Project Overview

Wine quality depends on several physicochemical characteristics such as acidity, sugar content, chlorides, sulfur dioxide, density, pH, sulphates, and alcohol.

This project explores the **Red Wine Quality dataset** and builds a machine learning model to learn the relationship between these chemical properties and wine quality.

The notebook includes:

* Data loading and exploration
* Dataset shape and information analysis
* Missing-value checking
* Exploratory Data Analysis (EDA)
* Feature and target separation
* Train-test splitting
* Random Forest model training
* Model evaluation using accuracy
* A basic predictive system for new wine measurements

The dataset contains **1,599 records and 12 columns**, including 11 physicochemical features and the `quality` target variable.

---

## 🎯 Objective

The main objective of this project is to:

> **Use physicochemical properties of red wine to build a machine learning model capable of predicting wine quality.**

The project demonstrates a complete beginner-friendly machine learning workflow, from data exploration to model prediction.

---

## 📊 Dataset

The project uses the `winequality-red.csv` dataset.

### Dataset Information

| Property          |     Value |
| ----------------- | --------: |
| Number of samples |     1,599 |
| Number of columns |        12 |
| Input features    |        11 |
| Target            | `quality` |
| Missing values    |         0 |

The notebook verifies that all 1,599 records contain non-null values across all 12 columns.

### Features

The dataset contains the following physicochemical features:

* `fixed acidity`
* `volatile acidity`
* `citric acid`
* `residual sugar`
* `chlorides`
* `free sulfur dioxide`
* `total sulfur dioxide`
* `density`
* `pH`
* `sulphates`
* `alcohol`

### Target

* `quality`

The feature matrix is created by removing the `quality` column, while `quality` is used as the target variable.

---

## 🔍 Exploratory Data Analysis

The notebook performs exploratory analysis to understand the distribution of wine quality and relationships between wine properties and quality.

Some of the visualizations include:

* Wine quality distribution
* Volatile acidity vs. quality
* Citric acid vs. quality
* Other feature-quality relationships

For example, the notebook uses a count plot to examine the distribution of the `quality` variable and a bar plot to analyze the relationship between volatile acidity and quality.

---

## 🧠 Machine Learning Workflow

The project follows this workflow:

```text
Wine Quality Dataset
        ↓
Data Loading
        ↓
Data Exploration
        ↓
Missing Value Check
        ↓
Exploratory Data Analysis
        ↓
Feature / Target Separation
        ↓
Train-Test Split
        ↓
Random Forest Classifier
        ↓
Prediction
        ↓
Accuracy Evaluation
```

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using an **80/20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This produces:

* **1,279 training samples**
* **320 testing samples**

as shown in the notebook.

---

## 🌲 Machine Learning Model

### Random Forest Classifier

The project uses `RandomForestClassifier` from Scikit-learn.

```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

The model uses the default Random Forest configuration and is trained on the wine physicochemical features.

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions. It is suitable for this type of tabular dataset because it can capture nonlinear relationships between the chemical properties and wine quality.

---

## 📈 Model Performance

The trained model is evaluated using **accuracy score**.

```python
test_accuracy = accuracy_score(y_pred, y_test)
print("Accuracy Score:", test_accuracy)
```

### Result

**Test Accuracy: 64.69%**

The notebook reports an accuracy of:

```text
0.646875
```

or approximately:

> **64.69% Accuracy**

---

## 🔮 Predictive System

The notebook also demonstrates prediction using a new set of physicochemical measurements.

Example input:

```python
input_data = [
    7.8,
    0.76,
    0.04,
    2.3,
    0.092,
    15.0,
    54.0,
    0.9970,
    3.26,
    0.65,
    9.8
]
```

The values are reshaped and passed to the trained model:

```python
input_data_as_array = np.asanyarray(input_data)

input_data_reshaped = input_data_as_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
```

The notebook then attempts to categorize the prediction as either:

```text
Good Quality Wine
```

or

```text
Bad Quality Wine
```

based on the predicted value.

---

## ⚠️ Current Implementation Note

The notebook contains two different target definitions:

```python
y = df["quality"]
```

and:

```python
Y = df["quality"].apply(lambda y_value: 1 if y_value >= 7 else 0)
```

However, the current Random Forest model is trained using `y`, which contains the original multi-class quality values rather than the binary `Y` variable.

Therefore, the predictive-system logic that checks:

```python
if prediction[0] == 1:
```

does not fully align with the model currently being trained.

### Recommended improvement

For a true **Good Quality vs Bad Quality** classification system, train the model using the binary target:

```python
y = df["quality"].apply(lambda x: 1 if x >= 7 else 0)
```

Then:

```python
model.fit(X_train, y_train)
```

and:

```python
if prediction[0] == 1:
    print("Good Quality Wine")
else:
    print("Bad Quality Wine")
```

This would make the model training and prediction logic consistent.

---

## 🛠️ Technologies Used

* **Python 3.13.6**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

The notebook imports NumPy, Pandas, Matplotlib, Seaborn, `StandardScaler`, `train_test_split`, `RandomForestClassifier`, and `accuracy_score`.

---

## 📁 Project Structure

```text
Wine-Quality/
│
├── Wine Quality.ipynb
├── winequality-red.csv
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Wine-Quality
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the notebook

Open:

```text
Wine Quality.ipynb
```

### 5. Run the cells

Run the notebook cells sequentially to:

1. Load the dataset
2. Explore the data
3. Perform EDA
4. Prepare the features and target
5. Split the dataset
6. Train the Random Forest model
7. Evaluate the model
8. Test the predictive system

---

## 📌 Key Takeaways

* The dataset contains **1,599 red wine samples**.
* There are **11 physicochemical input features**.
* The dataset contains **no missing values**.
* Exploratory analysis is performed using Matplotlib and Seaborn.
* A **Random Forest Classifier** is used for prediction.
* The current model achieves approximately **64.69% test accuracy**.
* The notebook also demonstrates prediction on new wine measurements.
* The binary quality target is defined in the notebook but is not currently used to train the Random Forest model.

---

## 🔮 Future Improvements

The project can be improved by:

* Correctly implementing the binary `Good/Bad` target.
* Comparing multiple machine learning algorithms.
* Performing feature scaling where appropriate.
* Handling class imbalance.
* Using confusion matrix and classification report.
* Performing hyperparameter tuning.
* Using cross-validation.
* Evaluating precision, recall, and F1-score.
* Saving the trained model with `joblib` or `pickle`.
* Building a web interface using Flask or Streamlit.
* Deploying the prediction system as an online application.

---

## 👨‍💻 Author

**Aryan Dongre**

Machine Learning / Data Science Project

---

## ⭐ Project Status

**Completed — Basic Machine Learning Implementation**

The current version demonstrates the complete basic workflow of loading wine-quality data, performing exploratory analysis, training a Random Forest model, evaluating its accuracy, and attempting predictions on new input data.
