# Forecasting Sticker Sales

**Kaggle Playground Series - Season 5, Episode 1**

## Overview

This repository contains my solution for the "Forecasting Sticker Sales" competition from the 2025 Kaggle Playground Series. The challenge involves predicting sticker sales across different countries using time series forecasting techniques.

> *"At Kaggle, we take stickers seriously!"™️*

## Competition Details

- **Competition Type**: Playground Prediction Competition
- **Duration**: January 1, 2025 - January 31, 2025
- **Participants**: 2,811 participants across 2,722 teams
- **Total Submissions**: 22,758

## Objective

The goal is to forecast the number of stickers sold (`num_sold`) for different countries based on historical sales data. This is a time series forecasting problem that requires understanding seasonal patterns, trends, and country-specific factors.

## Evaluation Metric

Submissions are evaluated using **Mean Absolute Percentage Error (MAPE)**:

```
MAPE = (1/n) * Σ|((Actual - Predicted) / Actual)| * 100
```

Where lower values indicate better performance.

## Dataset Information

- **Data Source**: Synthetically generated from real-world data
- **Type**: Time series data with sticker sales by country
- **Features**: [To be documented after data exploration]
- **Target Variable**: `num_sold` - number of stickers sold

### Submission Format

The submission file should contain predictions for each test set ID:

```csv
id,num_sold
230130,100
230131,100
230132,100
```

## Project Structure

```
forecasting-sticker-sales/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── models.py
├── submissions/
└── README.md
```

## Approach

### 1. Exploratory Data Analysis (EDA)
- Analyze temporal patterns and seasonality
- Examine country-specific sales trends
- Identify missing values and outliers
- Visualize sales distributions and correlations

### 2. Feature Engineering
- Create time-based features (day of week, month, quarter)
- Generate lag features and rolling statistics
- Country-specific encoding and aggregations
- Holiday and seasonal indicators

### 3. Modeling Strategy
- **Baseline Models**: Simple moving averages, linear regression
- **Traditional Time Series**: ARIMA, seasonal decomposition
- **Machine Learning**: Random Forest, XGBoost, LightGBM
- **Deep Learning**: LSTM, Transformer-based models
- **Ensemble Methods**: Combining multiple model predictions

### 4. Validation Strategy
- Time-based cross-validation to prevent data leakage
- Walk-forward validation for time series
- MAPE evaluation matching competition metric

## Key Insights

[To be updated after analysis]

## Results

- **Best Single Model**: [Model name and MAPE score]
- **Final Ensemble Score**: [MAPE score]
- **Competition Ranking**: [Final position]

## Dependencies

```python
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
xgboost>=1.6.0
lightgbm>=3.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
```

## Installation and Usage

1. Clone this repository:
```bash
git clone https://github.com/yourusername/forecasting-sticker-sales.git
cd forecasting-sticker-sales
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download competition data from Kaggle and place in `data/` directory

4. Run the analysis notebooks in order or execute the main pipeline:
```bash
python src/main.py
```

## About Kaggle Playground Series

The Tabular Playground Series provides the Kaggle community with lightweight challenges designed for learning and skill development. These competitions feature:

- Synthetically generated datasets based on real-world data
- Shorter competition durations (typically a few weeks)
- Focus on different aspects of ML and data science
- Opportunities for rapid iteration and experimentation

## Competition Tags

- **Beginner**: Suitable for newcomers to data science
- **Tabular**: Structured data analysis
- **Time Series Analysis**: Temporal pattern recognition
- **Mean Absolute Percentage Error**: Specific evaluation metric

## Acknowledgments

- **Competition Hosts**: Walter Reade and Elizabeth Park, Kaggle
- **Dataset**: Synthetically generated from real-world sticker sales data
- **Community**: Thanks to the Kaggle community for discussions and insights

## Citation

```
Walter Reade and Elizabeth Park. Forecasting Sticker Sales. 
https://kaggle.com/competitions/playground-series-s5e1, 2025. Kaggle.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Last updated: [Date]*