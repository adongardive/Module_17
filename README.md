# Bank Marketing Campaign Analysis

## Project Overview

This project analyzes data from a bank's marketing campaign to predict whether a client will subscribe to a term deposit. The goal is to use machine learning models to improve the bank’s marketing efficiency by targeting potential clients who are more likely to subscribe.

## Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Instances**: 41,188 rows
- **Features**: 20 (including client data, campaign information, and economic indicators)
- **Target Variable**: `y` (whether the client subscribed to a term deposit)

## Business Objective

The objective is to build a predictive model that can help the bank identify clients likely to subscribe to a term deposit, thereby improving conversion rates and optimizing marketing efforts.

## Methodology

1. **Data Exploration and Cleaning**
   - Handled missing values and irrelevant columns.
   - Visualized the data distributions.
   - Performed feature encoding for categorical variables and scaling for numerical features.

2. **Feature Engineering**
   - Created new features from the existing ones to improve model performance.
   - Focused on including economic indicators that might influence client behavior.

3. **Modeling**
   - Split the data into training and test sets (80/20 split).
   - Built and compared several models.
   - Performed hyperparameter tuning using Grid Search.

4. **Model Evaluation**
   - Evaluated models using various metrics:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - ROC AUC Score
   - Addressed class imbalance by adjusting class weights in the models.

## Results

- **Best Model**: Gradient Boosting
  - **ROC AUC Score**: 0.7751
  - **Precision**: 62.75%
  - **Recall**: 20.71%
  - **F1-Score**: 31.14%

The Gradient Boosting model was selected for its superior ability to distinguish between clients likely to subscribe and those who will not. Although recall is lower, it provides better precision, ensuring the bank can target likely subscribers effectively.

## Recommendations

- **Deploy the Gradient Boosting model** for targeting clients in future marketing campaigns.
- Regularly **monitor the model's performance** and update it with new data as needed.
- Consider further strategies to improve recall, such as resampling techniques or adjusting the classification threshold.

## How to Run the Project

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/adongardive/Module_17.git
   ```

2. **Install the Required Packages**:

   Use the following command to install the necessary libraries:

   ```bash
   pip install pandas==1.5.3 numpy==1.23.5 matplotlib==3.6.2 seaborn==0.12.1 scikit-learn==1.1.3 jupyter==1.0.0
   ```

3. **Download the Dataset**:
   - Download `bank-additional-full.csv` from the [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).
   - Place the dataset file into the project directory.

4. **Run the Jupyter Notebook**:

   ```bash
   jupyter notebook Bank_Marketing_Analysis.ipynb
   ```

5. **Execute All Cells**:
   - Open the notebook in your browser and run the cells to reproduce the analysis.

## Dependencies

- pandas==1.5.3
- numpy==1.23.5
- matplotlib==3.6.2
- seaborn==0.12.1
- scikit-learn==1.1.3
- jupyter==1.0.0
