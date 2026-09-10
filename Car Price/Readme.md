# 🚗 Car Price Prediction

A Machine Learning project that predicts the **selling price of a used car** based on various characteristics such as year, present price, kilometers driven, fuel type, seller type, transmission, and number of previous owners.

The project demonstrates the complete workflow of a basic regression problem, including data loading, exploratory analysis, categorical-data encoding, train-test splitting, model training, prediction, evaluation, and visualization.

---

## 📌 Project Overview

The objective of this project is to build a machine learning model that can estimate the selling price of a used car from its available features.

Two regression models are explored in the notebook:

* **Linear Regression**
* **Lasso Regression**

The models are trained using the car dataset and their predictions are evaluated using standard regression metrics.

---

## 📊 Dataset

The project uses the dataset:

```text
car data.csv
```

The dataset contains **301 records and 9 columns**.

### Dataset Features

| Feature         | Description                       |
| --------------- | --------------------------------- |
| `Car_Name`      | Name of the car                   |
| `Year`          | Manufacturing year                |
| `Selling_Price` | Selling price of the car — Target |
| `Present_Price` | Current/ex-showroom price         |
| `Kms_Driven`    | Kilometers driven                 |
| `Fuel_Type`     | Type of fuel used                 |
| `Seller_Type`   | Type of seller                    |
| `Transmission`  | Transmission type                 |
| `Owner`         | Number of previous owners         |

The notebook checks for missing values, and all columns contain **0 missing values**.

---

## 🔄 Data Preprocessing

### 1. Categorical Data Encoding

The categorical columns are converted into numerical values so that they can be used by the machine learning models.

#### Fuel Type

```text
Petrol  → 0
Diesel  → 1
CNG     → 2
```

#### Seller Type

```text
Dealer      → 0
Individual  → 1
```

#### Transmission

```text
Manual      → 0
Automatic   → 1
```

These transformations are implemented directly in the notebook.

---

## 🎯 Feature Selection

`Car_Name` is removed from the input features, while `Selling_Price` is used as the target variable.

```python
X = car_data.drop(['Car_Name', 'Selling_Price'], axis=1)
y = car_data["Selling_Price"]
```

The model therefore uses:

```text
Year
Present_Price
Kms_Driven
Fuel_Type
Seller_Type
Transmission
Owner
```

as input features.

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This results in:

* **240 training samples**
* **61 testing samples**

---

## 🤖 Machine Learning Models

### 1. Linear Regression

The first model used is **Linear Regression**.

```python
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

The trained model is then used to predict selling prices for the test dataset.

### 2. Lasso Regression

The project also implements **Lasso Regression**.

```python
lasso_model = Lasso()

lasso_model.fit(X_train, y_train)

y_lasso_pred = lasso_model.predict(X_test)
```

The Lasso model generates predictions for all 61 test samples.

---

## 📈 Model Evaluation

The Linear Regression model is evaluated using four regression metrics:

| Metric       |  Score |
| ------------ | -----: |
| **MAE**      | 1.2218 |
| **MSE**      | 3.5289 |
| **RMSE**     | 1.8785 |
| **R² Score** | 0.8468 |

The **R² score of approximately 0.847** indicates that the Linear Regression model explains a substantial portion of the variation in the target selling prices on the test set.

> **Note:** The notebook implements Lasso Regression, but a separate set of Lasso evaluation metrics is not recorded in the provided notebook output.

---

## 📊 Visualization

The project visualizes the relationship between:

* **Actual Prices**
* **Predicted Prices**

using a scatter plot.

```python
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
```

This visualization helps assess how closely the predicted prices follow the actual selling prices.

---

## 🔬 Project Workflow

```text
Dataset
   ↓
Load Data
   ↓
Data Inspection
   ↓
Check Missing Values
   ↓
Encode Categorical Features
   ↓
Separate Features & Target
   ↓
Train-Test Split
   ↓
Linear Regression
   ↓
Price Prediction
   ↓
Model Evaluation
   ↓
Visualization
   ↓
Lasso Regression
   ↓
Price Prediction
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data manipulation
* **NumPy** — Numerical operations
* **Matplotlib** — Data visualization
* **Seaborn** — Data visualization
* **Scikit-learn** — Machine learning models and evaluation
* **Jupyter Notebook** — Development environment

The notebook imports `pandas`, `matplotlib`, `seaborn`, `LinearRegression`, `Lasso`, `train_test_split`, and regression evaluation metrics from scikit-learn.

---

## 📁 Project Structure

```text
Car-Price-Prediction/
│
├── Car Price.ipynb
├── car data.csv
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd Car-Price-Prediction
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
Car Price.ipynb
```

Run the notebook cells sequentially to reproduce the analysis and model results.

---

## 📌 Key Learnings

Through this project, the following Machine Learning concepts are demonstrated:

* Loading and exploring a dataset
* Checking dataset dimensions
* Handling categorical variables
* Feature and target separation
* Train-test splitting
* Linear Regression
* Lasso Regression
* Making predictions
* MAE, MSE, RMSE and R² evaluation
* Actual vs predicted price visualization

---

## 🔮 Future Improvements

Possible improvements for this project include:

* Compare additional regression algorithms
* Perform feature scaling where appropriate
* Perform hyperparameter tuning
* Add cross-validation
* Improve categorical feature encoding
* Compare Linear Regression and Lasso using the same evaluation metrics
* Build an interactive web application for price prediction
* Deploy the trained model as an API

---

## ⚠️ Disclaimer

This project is created for **educational and learning purposes**. The predicted prices should not be considered professional or official vehicle valuations.

---

## 👨‍💻 Author

**Aryan Dongre**

This project is part of my Machine Learning practice and demonstrates the implementation of regression techniques for a real-world price prediction problem.
