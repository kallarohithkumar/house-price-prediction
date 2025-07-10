# House Price Prediction

This project implements a **Linear Regression** model using **Python** and **scikit-learn** to predict house prices based on features such as:

- Bedrooms
- Bathrooms
- Living Area (sqft)
- Lot Area (sqft)
- Floors
- Waterfront
- View
- Condition

## 📁 Files

- `house_price_prediction.py` – The main script with model training, prediction, evaluation, and plots.
- `doc.csv` – The sample dataset used for training/testing.

## 📊 Outputs

- Correlation Heatmap
- Actual vs Predicted Price Scatter Plot
- Residual Plot
- Feature Importance (Bar Plot)
- R-squared Value
- Predicted Price for Sample Input

## 🧠 Libraries Used

- pandas
- matplotlib
- seaborn
- scikit-learn

## 💻 How to Run

1. Clone the repo or upload files to [Google Colab](https://colab.research.google.com)
2. Upload `doc.csv`
3. Run `house_price_prediction.py` line-by-line

## 📈 Sample Prediction

For input: `[3, 2, 1500, 4000, 1, 0, 0, 3]`  
**Predicted Price:** _(based on model output)_
