# Complete Solution: Forecasting Sticker Sales
# Kaggle Playground Series - Season 5, Episode 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# ================================
# STEP 1: DATA LOADING AND EXPLORATION
# ================================

def load_and_explore_data():
    """Load the competition data and perform initial exploration"""
    print("Loading data...")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    sample_sub = pd.read_csv('data/sample_submission.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Sample submission shape: {sample_sub.shape}")

    print("\n=== TRAIN DATA INFO ===")
    print(train.info())
    print("\n=== TRAIN DATA HEAD ===")
    print(train.head())

    print("\n=== TRAIN DATA DESCRIPTION ===")
    print(train.describe())

    print("\n=== MISSING VALUES ===")
    print(train.isnull().sum())

    return train, test, sample_sub


# ================================
# STEP 2: DATA PREPROCESSING
# ================================

def preprocess_data(train, test):
    """Clean and preprocess the data"""
    print("Preprocessing data...")

    # Combine train and test for consistent preprocessing
    train['is_train'] = 1
    test['is_train'] = 0
    test['num_sold'] = np.nan  # Add target column to test

    df = pd.concat([train, test], ignore_index=True)

    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'country', 'store', 'product'])

    # Handle missing values in numerical columns
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        elif df[col].dtype == 'object' and col not in ['country', 'store', 'product', 'date']:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Fill missing values in categorical columns
    categorical_cols = ['country', 'store', 'product']
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna('Unknown', inplace=True)

    print(f"Data shape after preprocessing: {df.shape}")
    print(f"Missing values after preprocessing: {df.isnull().sum().sum()}")

    return df


# ================================
# STEP 3: FEATURE ENGINEERING
# ================================

def create_time_features(df, date_col='date'):
    """Create time-based features"""
    print("Creating time features...")

    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df['week_of_year'] = df[date_col].dt.isocalendar().week

    # Cyclical encoding for circular features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    return df


def create_lag_features(df, target_col, group_cols=['country', 'store', 'product'], lags=[1, 2, 3, 7, 14, 30]):
    """Create lag features for time series"""
    print("Creating lag features...")

    # Filter group_cols to only include columns that exist
    group_cols = [col for col in group_cols if col in df.columns]

    if group_cols:
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
    else:
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    return df


def create_rolling_features(df, target_col, group_cols=['country', 'store', 'product'], windows=[3, 7, 14, 30]):
    """Create rolling window features"""
    print("Creating rolling features...")

    # Filter group_cols to only include columns that exist
    group_cols = [col for col in group_cols if col in df.columns]

    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    if group_cols:
        # Sort by group columns and date to ensure proper ordering
        if 'date' in df_copy.columns:
            df_copy = df_copy.sort_values(group_cols + ['date'])
        else:
            df_copy = df_copy.sort_values(group_cols)

        for window in windows:
            print(f"Creating rolling features for window {window}...")

            # Method 1: Using transform (recommended)
            df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df_copy[f'{target_col}_rolling_std_{window}'] = df_copy.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df_copy[f'{target_col}_rolling_min_{window}'] = df_copy.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
            df_copy[f'{target_col}_rolling_max_{window}'] = df_copy.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )

    else:
        # No grouping - simple rolling on entire series
        for window in windows:
            df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy[target_col].rolling(window, min_periods=1).mean()
            df_copy[f'{target_col}_rolling_std_{window}'] = df_copy[target_col].rolling(window, min_periods=1).std()
            df_copy[f'{target_col}_rolling_min_{window}'] = df_copy[target_col].rolling(window, min_periods=1).min()
            df_copy[f'{target_col}_rolling_max_{window}'] = df_copy[target_col].rolling(window, min_periods=1).max()

    return df_copy


# Alternative implementation using a more robust approach
def create_rolling_features_alternative(df, target_col, group_cols=['country', 'store', 'product'],
                                        windows=[3, 7, 14, 30]):
    """Alternative implementation using apply and concat"""
    print("Creating rolling features (alternative method)...")

    # Filter group_cols to only include columns that exist
    group_cols = [col for col in group_cols if col in df.columns]

    df_copy = df.copy()

    if group_cols:
        # Sort by group columns and date
        if 'date' in df_copy.columns:
            df_copy = df_copy.sort_values(group_cols + ['date'])
        else:
            df_copy = df_copy.sort_values(group_cols)

        for window in windows:
            print(f"Creating rolling features for window {window}...")

            # Create rolling features for each group separately
            def calculate_rolling_stats(group):
                rolling_obj = group[target_col].rolling(window, min_periods=1)
                return pd.DataFrame({
                    f'{target_col}_rolling_mean_{window}': rolling_obj.mean(),
                    f'{target_col}_rolling_std_{window}': rolling_obj.std(),
                    f'{target_col}_rolling_min_{window}': rolling_obj.min(),
                    f'{target_col}_rolling_max_{window}': rolling_obj.max()
                }, index=group.index)

            # Apply the function to each group and concatenate results
            rolling_features = df_copy.groupby(group_cols).apply(calculate_rolling_stats)

            # Reset index to match original dataframe
            rolling_features = rolling_features.droplevel(list(range(len(group_cols))))
            rolling_features = rolling_features.reindex(df_copy.index)

            # Add the features to the main dataframe
            for col in rolling_features.columns:
                df_copy[col] = rolling_features[col]

    else:
        # No grouping - simple rolling on entire series
        for window in windows:
            df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy[target_col].rolling(window, min_periods=1).mean()
            df_copy[f'{target_col}_rolling_std_{window}'] = df_copy[target_col].rolling(window, min_periods=1).std()
            df_copy[f'{target_col}_rolling_min_{window}'] = df_copy[target_col].rolling(window, min_periods=1).min()
            df_copy[f'{target_col}_rolling_max_{window}'] = df_copy[target_col].rolling(window, min_periods=1).max()

    return df_copy


# If you still have issues, here's a simpler version that processes one window at a time
def create_rolling_features_simple(df, target_col, group_cols=['country', 'store', 'product'], windows=[3, 7, 14, 30]):
    """Simplified version that handles one feature at a time"""
    print("Creating rolling features (simple method)...")

    # Filter group_cols to only include columns that exist
    group_cols = [col for col in group_cols if col in df.columns]

    df_result = df.copy()

    # Ensure proper sorting
    if 'date' in df_result.columns:
        df_result = df_result.sort_values(group_cols + ['date'] if group_cols else ['date'])
    elif group_cols:
        df_result = df_result.sort_values(group_cols)

    for window in windows:
        print(f"Processing window {window}...")

        if group_cols:
            # Use transform to avoid index issues
            grouped = df_result.groupby(group_cols)[target_col]

            # Calculate each rolling feature separately
            df_result[f'{target_col}_rolling_mean_{window}'] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).mean())

            df_result[f'{target_col}_rolling_std_{window}'] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).std())

            df_result[f'{target_col}_rolling_min_{window}'] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).min())

            df_result[f'{target_col}_rolling_max_{window}'] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).max())
        else:
            # Simple case without grouping
            rolling = df_result[target_col].rolling(window, min_periods=1)
            df_result[f'{target_col}_rolling_mean_{window}'] = rolling.mean()
            df_result[f'{target_col}_rolling_std_{window}'] = rolling.std()
            df_result[f'{target_col}_rolling_min_{window}'] = rolling.min()
            df_result[f'{target_col}_rolling_max_{window}'] = rolling.max()

    return df_result


def create_categorical_features(df):
    """Create and encode categorical features"""
    print("Creating categorical features...")

    # Identify categorical columns
    categorical_cols = ['country', 'store', 'product']
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    train_data = df[df['is_train'] == 1]

    for col in categorical_cols:
        print(f"Encoding {col}...")

        # Target encoding (mean of target for each category)
        if not train_data.empty:
            target_means = train_data.groupby(col)['num_sold'].mean()
            df[f'{col}_target_encoded'] = df[col].map(target_means)
            df[f'{col}_target_encoded'].fillna(train_data['num_sold'].mean(), inplace=True)

        # Frequency encoding
        category_counts = df[col].value_counts()
        df[f'{col}_frequency'] = df[col].map(category_counts)

        # Count encoding (number of unique values per category in train)
        category_nunique = train_data.groupby(col)['num_sold'].count()
        df[f'{col}_count'] = df[col].map(category_nunique)
        df[f'{col}_count'].fillna(0, inplace=True)

        # Label encoding for tree-based models
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[f'{col}_label_encoded'] = le.fit_transform(df[col].astype(str))

    return df


def feature_engineering_pipeline(df):
    """Complete feature engineering pipeline"""
    print("Starting feature engineering pipeline...")

    # Identify date and target columns
    target_col = 'num_sold'

    # Create time features
    if 'date' in df.columns:
        df = create_time_features(df, 'date')

    # Create categorical features first (before lag features)
    df = create_categorical_features(df)

    # Create lag and rolling features (only on training data initially)
    train_mask = df['is_train'] == 1
    if train_mask.sum() > 0:
        # Use the grouping columns that exist
        group_cols = ['country', 'store', 'product']
        group_cols = [col for col in group_cols if col in df.columns]

        df = create_lag_features(df, target_col, group_cols)
        df = create_rolling_features(df, target_col, group_cols)

    return df


# ================================
# STEP 4: VISUALIZATION
# ================================

def create_visualizations(df):
    """Create exploratory visualizations"""
    print("Creating visualizations...")

    train_data = df[df['is_train'] == 1].copy()

    plt.figure(figsize=(15, 12))

    # 1. Target distribution
    plt.subplot(2, 3, 1)
    plt.hist(train_data['num_sold'], bins=50, alpha=0.7)
    plt.title('Distribution of Sticker Sales')
    plt.xlabel('Number Sold')
    plt.ylabel('Frequency')

    # 2. Sales over time (if date column exists)
    if 'date' in df.columns:
        plt.subplot(2, 3, 2)
        daily_sales = train_data.groupby('date')['num_sold'].sum()
        plt.plot(daily_sales.index, daily_sales.values)
        plt.title('Total Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)

    # 3. Sales by month
    plt.subplot(2, 3, 3)
    monthly_sales = train_data.groupby('month')['num_sold'].mean()
    plt.bar(monthly_sales.index, monthly_sales.values)
    plt.title('Average Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')

    # 4. Sales by day of week
    plt.subplot(2, 3, 4)
    dow_sales = train_data.groupby('dayofweek')['num_sold'].mean()
    plt.bar(dow_sales.index, dow_sales.values)
    plt.title('Average Sales by Day of Week')
    plt.xlabel('Day of Week (0=Monday)')
    plt.ylabel('Average Sales')

    # 5. Sales by country (if exists)
    if 'country' in df.columns:
        plt.subplot(2, 3, 5)
        country_sales = train_data.groupby('country')['num_sold'].mean().sort_values(ascending=False)
        top_countries = country_sales.head(10)
        plt.bar(range(len(top_countries)), top_countries.values)
        plt.title('Top 10 Countries by Average Sales')
        plt.xlabel('Country')
        plt.ylabel('Average Sales')
        plt.xticks(range(len(top_countries)), top_countries.index, rotation=45)

    # 6. Correlation matrix
    plt.subplot(2, 3, 6)
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    corr_matrix = train_data[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')

    plt.tight_layout()
    plt.show()


# ================================
# STEP 5: MODEL TRAINING
# ================================

def prepare_model_data(df):
    """Prepare data for modeling"""
    print("Preparing model data...")

    # Split back to train and test
    train_data = df[df['is_train'] == 1].copy()
    test_data = df[df['is_train'] == 0].copy()

    # Remove non-feature columns (including original categorical columns)
    exclude_cols = [
        'is_train', 'num_sold', 'id', 'date',  # Basic columns to exclude
        'country', 'store', 'product'  # Original categorical columns (we use encoded versions)
    ]
    exclude_cols = [col for col in exclude_cols if col in df.columns]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X_train = train_data[feature_cols]
    y_train = train_data['num_sold']
    X_test = test_data[feature_cols]

    # Handle remaining missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    print(f"Feature columns: {len(feature_cols)}")
    print(f"Sample feature columns: {feature_cols[:10]}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Verify all columns are numeric
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            print(f"Warning: Column {col} is still object type!")

    return X_train, y_train, X_test, feature_cols


def train_lightgbm(X_train, y_train, X_val=None, y_val=None):
    """Train LightGBM model"""
    print("Training LightGBM...")

    params = {
        'objective': 'regression',
        'metric': 'mape',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(params, train_data, valid_sets=[val_data],
                          num_boost_round=1000, callbacks=[lgb.early_stopping(100)])
    else:
        model = lgb.train(params, train_data, num_boost_round=500)

    return model


def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """Train XGBoost model"""
    print("Training XGBoost...")

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mape',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)

    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(params, dtrain, evals=[(dval, 'val')],
                          num_boost_round=1000, early_stopping_rounds=100, verbose_eval=False)
    else:
        model = xgb.train(params, dtrain, num_boost_round=500)

    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


# ================================
# STEP 6: MODEL VALIDATION
# ================================

def time_series_cv(X, y, n_splits=5):
    """Perform time series cross-validation"""
    print("Performing time series cross-validation...")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1}/{n_splits}")

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Train models
        lgb_model = train_lightgbm(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        xgb_model = train_xgboost(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        rf_model = train_random_forest(X_train_fold, y_train_fold)

        # Make predictions
        lgb_pred = lgb_model.predict(X_val_fold, num_iteration=lgb_model.best_iteration)
        xgb_pred = xgb_model.predict(xgb.DMatrix(X_val_fold))
        rf_pred = rf_model.predict(X_val_fold)

        # Ensemble prediction
        ensemble_pred = 0.4 * lgb_pred + 0.4 * xgb_pred + 0.2 * rf_pred

        # Calculate MAPE
        mape = mean_absolute_percentage_error(y_val_fold, ensemble_pred)
        cv_scores.append(mape)
        print(f"Fold {fold + 1} MAPE: {mape:.6f}")

    print(f"\nCross-validation MAPE: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores) * 2:.6f})")
    return cv_scores


# ================================
# STEP 7: FINAL MODEL AND PREDICTIONS
# ================================

def train_final_models(X_train, y_train):
    """Train final models on all training data"""
    print("Training final models...")

    lgb_model = train_lightgbm(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    return lgb_model, xgb_model, rf_model


def make_predictions(models, X_test):
    """Make final predictions"""
    print("Making predictions...")

    lgb_model, xgb_model, rf_model = models

    lgb_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    xgb_pred = xgb_model.predict(xgb.DMatrix(X_test))
    rf_pred = rf_model.predict(X_test)

    # Ensemble prediction
    ensemble_pred = 0.4 * lgb_pred + 0.4 * xgb_pred + 0.2 * rf_pred

    return ensemble_pred


def create_submission(test_ids, predictions, filename='submission.csv'):
    """Create submission file"""
    print(f"Creating submission file: {filename}")

    submission = pd.DataFrame({
        'id': test_ids,
        'num_sold': predictions
    })

    submission.to_csv(filename, index=False)
    print(f"Submission saved with {len(submission)} predictions")
    return submission


# ================================
# STEP 8: MAIN EXECUTION PIPELINE
# ================================

def main():
    """Main execution pipeline"""
    print("=== STICKER SALES FORECASTING PIPELINE ===\n")

    # Step 1: Load and explore data
    train, test, sample_sub = load_and_explore_data()

    # Step 2: Preprocess data
    df = preprocess_data(train, test)

    # Step 3: Feature engineering
    df = feature_engineering_pipeline(df)

    # Step 4: Create visualizations
    create_visualizations(df)

    # Step 5: Prepare model data
    X_train, y_train, X_test, feature_cols = prepare_model_data(df)

    print(f"Final feature count: {len(feature_cols)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Step 6: Cross-validation
    cv_scores = time_series_cv(X_train, y_train)

    # Step 7: Train final models
    final_models = train_final_models(X_train, y_train)

    # Step 8: Make predictions
    predictions = make_predictions(final_models, X_test)

    # Step 9: Create submission
    test_ids = test['id'] if 'id' in test.columns else range(len(test))
    submission = create_submission(test_ids, predictions)

    print("\n=== PIPELINE COMPLETE ===")
    print(f"Final CV MAPE: {np.mean(cv_scores):.6f}")
    print("Submission file created: submission.csv")

    return submission, cv_scores


# ================================
# ADDITIONAL UTILITY FUNCTIONS
# ================================

def feature_importance_analysis(models, feature_cols):
    """Analyze feature importance"""
    print("Analyzing feature importance...")

    lgb_model, xgb_model, rf_model = models

    # Get feature importances
    lgb_importance = lgb_model.feature_importance(importance_type='gain')
    rf_importance = rf_model.feature_importances_

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'lgb_importance': lgb_importance,
        'rf_importance': rf_importance
    })

    importance_df['avg_importance'] = (importance_df['lgb_importance'] + importance_df['rf_importance']) / 2
    importance_df = importance_df.sort_values('avg_importance', ascending=False)

    # Plot top features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['avg_importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Average Feature Importance')
    plt.title('Top 20 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return importance_df


def hyperparameter_tuning():
    """Example of hyperparameter tuning (optional)"""
    print("Hyperparameter tuning example...")

    # LightGBM parameter grid
    lgb_params = {
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1],
        'feature_fraction': [0.8, 0.9, 1.0],
        'bagging_fraction': [0.8, 0.9, 1.0]
    }

    # XGBoost parameter grid
    xgb_params = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    print("Use GridSearchCV or RandomizedSearchCV with time series split for tuning")
    return lgb_params, xgb_params


# ================================
# RUN THE PIPELINE
# ================================

if __name__ == "__main__":
    # Execute the main pipeline
    submission, cv_scores = main()

    # Optional: Run feature importance analysis
    # importance_df = feature_importance_analysis(final_models, feature_cols)