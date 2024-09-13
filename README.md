# Gold Price Prediction Using Machine Learning

This project demonstrates the prediction of gold prices using a machine learning model (Random Forest Regressor). The model is trained on historical gold price data to predict future values with a focus on correlation analysis and error measurement (R squared).

## Dataset
The dataset used contains historical data of gold prices and related financial parameters, loaded from a CSV file.
The date approximation is of around 10 years of data.

## Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Data visualization
- **seaborn**: Advanced data visualization
- **scikit-learn**: Machine learning models and metrics

## Steps Involved
1. **Data Loading**: The dataset is loaded using `pandas` from a CSV file.
2. **Data Cleaning**: Checked for missing values and converted columns to numeric types.
3. **Exploratory Data Analysis (EDA)**:
    - Used correlation matrix to find relationships between variables.
    - Visualized the distribution of gold prices.
4. **Model Training**:
    - Features (`X`) were extracted by removing the `Date` and `GLD` columns.
    - The target (`Y`) is the `GLD` (gold price).
    - The data is split into training and testing sets using an 80/20 split.
5. **Prediction**:
    - The Random Forest Regressor model is trained on the training set.
    - The model predicts the values on the test set, and the performance is evaluated using the R-squared error.
6. **Visualization**:
    - Comparison of actual vs predicted gold prices using a line plot.

## Model Performance
- The model's accuracy is measured using the R-squared error, where a higher score indicates better performance.

## Visualizations
- **Correlation Heatmap**: Shows the relationships between different financial factors and gold prices.
- **Actual vs Predicted Plot**: Compares the model's predictions with actual values.


## Requirements
- Python 3.9.6 or above
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
