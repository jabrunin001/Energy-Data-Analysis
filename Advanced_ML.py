import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and preprocess data
def load_data(filepath):
    """
    Load energy consumption data from CSV file
    """
    df = pd.read_csv(filepath)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Create binary peak hours feature (typically 2pm-6pm)
    df['is_peak_hour'] = df['hour'].between(14, 18).astype(int)
    
    # Calculate squared terms for temperature (for non-linear relationships)
    df['temperature_squared'] = df['temperature'] ** 2
    
    # Create interaction terms
    df['temp_peak_interaction'] = df['temperature'] * df['is_peak_hour']
    
    return df

# Sample code for simulating data when CSV isn't available
def generate_synthetic_data(num_regions=5, days=60):
    """
    Generate synthetic energy consumption data
    """
    data = []
    regions = [f'Region_{i+1}' for i in range(num_regions)]
    
    for day in range(days):
        for hour in range(24):
            timestamp = pd.Timestamp('2023-01-01') + pd.Timedelta(days=day) + pd.Timedelta(hours=hour)
            
            for region in regions:
                region_idx = regions.index(region)
                
                # Base consumption with daily cycle
                base_consumption = 1000 + 500 * np.sin((hour - 6) * np.pi / 12)
                
                # Weekend effect
                if timestamp.dayofweek >= 5:  # Saturday or Sunday
                    base_consumption *= 0.8
                
                # Seasonal effect
                month_effect = np.sin((day / 30) * 2 * np.pi)
                base_consumption += month_effect * 200
                
                # Regional variation
                region_multiplier = 0.8 + (region_idx * 0.1)
                base_consumption *= region_multiplier
                
                # Random noise
                base_consumption += 100 * (np.random.random() - 0.5)
                
                # Temperature
                base_temp = 60 + 20 * np.sin((day / 365) * 2 * np.pi)
                hourly_temp_variation = 10 * np.sin((hour - 14) * np.pi / 12)
                temperature = base_temp + hourly_temp_variation + 5 * (np.random.random() - 0.5)
                
                # Business activity
                business_activity = base_consumption / 1000 * (0.8 + 0.4 * np.random.random())
                
                # Renewable percentage
                renewable_value = 20 + region_idx * 3 + day / 10 + 10 * (np.random.random() - 0.5)
                renewable_percentage = np.clip(renewable_value, 5, 75)
                
                # Hospital admissions
                admission_base = 100
                if temperature > 85 or temperature < 30:
                    admission_base += abs(temperature - 75) / 5
                hospital_admissions = round(admission_base * (0.8 + 0.4 * np.random.random()))
                
                data.append({
                    'region': region,
                    'timestamp': timestamp,
                    'consumption': base_consumption,
                    'temperature': temperature,
                    'business_activity': business_activity,
                    'renewable_percentage': renewable_percentage,
                    'hospital_admissions': hospital_admissions
                })
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_peak_hour'] = df['hour'].between(14, 18).astype(int)
    df['temperature_squared'] = df['temperature'] ** 2
    df['temp_peak_interaction'] = df['temperature'] * df['is_peak_hour']
    
    return df

# 2. Exploratory Data Analysis
def perform_eda(df):
    """
    Perform exploratory data analysis on the dataset
    """
    print(f"Dataset shape: {df.shape}")
    print("\nFeature summary statistics:")
    print(df.describe())
    
    # Correlation analysis
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Energy Consumption Features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    # Distribution of consumption by region
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='region', y='consumption', data=df)
    plt.title('Energy Consumption Distribution by Region')
    plt.tight_layout()
    plt.savefig('consumption_by_region.png')
    
    # Time series plot for a single region
    region1_data = df[df['region'] == 'Region_1'].sort_values('timestamp')
    plt.figure(figsize=(15, 6))
    plt.plot(region1_data['timestamp'], region1_data['consumption'])
    plt.title('Energy Consumption Time Series for Region_1')
    plt.xlabel('Timestamp')
    plt.ylabel('Consumption')
    plt.tight_layout()
    plt.savefig('region1_timeseries.png')
    
    # Consumption by hour of day and day of week
    pivot_hour_day = df.pivot_table(
        values='consumption', 
        index='hour',
        columns='day_of_week',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_hour_day, cmap='viridis', annot=False)
    plt.title('Average Consumption by Hour and Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Hour of Day')
    plt.tight_layout()
    plt.savefig('consumption_hour_day_heatmap.png')
    
    return corr_matrix

# 3. Time series forecasting models
def time_series_forecast(df, region, forecast_days=30):
    """
    Build and evaluate time series forecasting models
    """
    # Filter data for the specified region
    region_data = df[df['region'] == region].copy()
    
    # Aggregate to daily consumption for simpler forecasting
    daily_data = region_data.groupby(region_data['timestamp'].dt.date)['consumption'].mean().reset_index()
    daily_data.rename(columns={'timestamp': 'ds', 'consumption': 'y'}, inplace=True)
    
    # Train-test split (use last 20% of data for testing)
    train_size = int(len(daily_data) * 0.8)
    train_data = daily_data[:train_size]
    test_data = daily_data[train_size:]
    
    # 1. Prophet model
    model_prophet = Prophet(
        yearly_seasonality=False,  # Our data might not span a year
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model_prophet.fit(train_data)
    
    # Create future dataframe for prediction
    future = model_prophet.make_future_dataframe(periods=len(test_data) + forecast_days)
    forecast = model_prophet.predict(future)
    
    # Calculate metrics on test set
    prophet_predictions = forecast[train_size:train_size+len(test_data)]['yhat'].values
    prophet_rmse = np.sqrt(mean_squared_error(test_data['y'], prophet_predictions))
    prophet_mae = mean_absolute_error(test_data['y'], prophet_predictions)
    
    print(f"Prophet Model for {region}:")
    print(f"RMSE: {prophet_rmse:.2f}")
    print(f"MAE: {prophet_mae:.2f}")
    
    # 2. ARIMA model (using statsmodels)
    try:
        # Fit ARIMA model (p,d,q) = (5,1,1) - these would normally be tuned
        model_arima = ARIMA(train_data['y'], order=(5,1,1))
        arima_results = model_arima.fit()
        
        # Generate predictions
        arima_predictions = arima_results.forecast(steps=len(test_data))
        
        arima_rmse = np.sqrt(mean_squared_error(test_data['y'], arima_predictions))
        arima_mae = mean_absolute_error(test_data['y'], arima_predictions)
        
        print(f"\nARIMA Model for {region}:")
        print(f"RMSE: {arima_rmse:.2f}")
        print(f"MAE: {arima_mae:.2f}")
    except:
        print("ARIMA model fitting failed. This can happen with certain data patterns.")
        arima_predictions = None
        arima_rmse = None
    
    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(daily_data['ds'], daily_data['y'], label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Forecast', color='blue')
    
    if arima_predictions is not None:
        full_dates = list(daily_data['ds']) + [daily_data['ds'].iloc[-1] + pd.Timedelta(days=i+1) for i in range(forecast_days)]
        arima_full_predictions = list(arima_results.fittedvalues) + list(arima_predictions)
        plt.plot(full_dates[-len(arima_full_predictions):], arima_full_predictions, label='ARIMA Forecast', color='green')
    
    plt.title(f'Energy Consumption Forecast for {region}')
    plt.xlabel('Date')
    plt.ylabel('Consumption')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{region}_forecast.png')
    
    # Return the forecast for future use
    return {
        'prophet_forecast': forecast,
        'prophet_rmse': prophet_rmse,
        'prophet_mae': prophet_mae,
        'arima_rmse': arima_rmse
    }

# 4. Consumption Prediction Model
def build_consumption_model(df):
    """
    Build and evaluate machine learning models to predict energy consumption
    """
    # Prepare features and target
    features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour', 
                'temperature', 'temperature_squared', 'renewable_percentage']
    
    # Create dummy variables for region
    df_with_dummies = pd.get_dummies(df, columns=['region'], drop_first=False)
    region_cols = [col for col in df_with_dummies.columns if col.startswith('region_')]
    features.extend(region_cols)
    
    X = df_with_dummies[features]
    y = df_with_dummies['consumption']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. RandomForest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    rf_preds = rf_model.predict(X_test)
    
    # Evaluate the model
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    
    print("\nRandom Forest Model for Consumption Prediction:")
    print(f"RMSE: {rf_rmse:.2f}")
    print(f"MAE: {rf_mae:.2f}")
    print(f"R² Score: {rf_r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))
    
    # 2. Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    
    # Make predictions
    gb_preds = gb_model.predict(X_test)
    
    # Evaluate the model
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_preds))
    gb_mae = mean_absolute_error(y_test, gb_preds)
    gb_r2 = r2_score(y_test, gb_preds)
    
    print("\nGradient Boosting Model for Consumption Prediction:")
    print(f"RMSE: {gb_rmse:.2f}")
    print(f"MAE: {gb_mae:.2f}")
    print(f"R² Score: {gb_r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, gb_preds, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Consumption')
    plt.ylabel('Predicted Consumption')
    plt.title('Gradient Boosting: Actual vs Predicted Consumption')
    plt.tight_layout()
    plt.savefig('consumption_prediction.png')
    
    return {
        'rf_model': rf_model,
        'rf_rmse': rf_rmse,
        'rf_r2': rf_r2,
        'gb_model': gb_model,
        'gb_rmse': gb_rmse,
        'gb_r2': gb_r2,
        'feature_importance': feature_importance
    }

# 5. Health Impact Analysis Model
def health_impact_model(df):
    """
    Build model to predict health impacts based on energy and environmental factors
    """
    # Prepare features
    features = ['consumption', 'temperature', 'temperature_squared', 'renewable_percentage',
                'hour', 'is_weekend', 'is_peak_hour']
    
    # Add region dummies
    df_with_dummies = pd.get_dummies(df, columns=['region'], drop_first=False)
    region_cols = [col for col in df_with_dummies.columns if col.startswith('region_')]
    features.extend(region_cols)
    
    X = df_with_dummies[features]
    y = df_with_dummies['hospital_admissions']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    rf_preds = rf_model.predict(X_test)
    
    # Evaluate
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    
    print("\nRandom Forest Model for Health Impact Prediction:")
    print(f"RMSE: {rf_rmse:.2f}")
    print(f"MAE: {rf_mae:.2f}")
    print(f"R² Score: {rf_r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop Health Impact Factors:")
    print(feature_importance.head(10))
    
    # Scenario Analysis: Impact of reducing peak consumption by 10%
    # Create a copy of test data and reduce peak consumption by 10%
    X_test_reduced = X_test.copy()
    peak_hours_mask = X_test_reduced['is_peak_hour'] == 1
    X_test_reduced.loc[peak_hours_mask, 'consumption'] *= 0.9
    
    # Predict with reduced consumption
    preds_reduced = rf_model.predict(X_test_reduced)
    
    # Calculate average reduction in hospital admissions
    avg_original = rf_preds.mean()
    avg_reduced = preds_reduced.mean()
    percent_reduction = ((avg_original - avg_reduced) / avg_original) * 100
    
    print(f"\nImpact of 10% Peak Consumption Reduction:")
    print(f"Current Average Daily Admissions: {avg_original:.2f}")
    print(f"Projected Average Daily Admissions: {avg_reduced:.2f}")
    print(f"Percent Reduction: {percent_reduction:.2f}%")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Factors Influencing Hospital Admissions')
    plt.tight_layout()
    plt.savefig('health_impact_factors.png')
    
    return {
        'model': rf_model,
        'rmse': rf_rmse,
        'r2': rf_r2,
        'feature_importance': feature_importance,
        'percent_reduction': percent_reduction
    }

# 6. Anomaly Detection
def detect_anomalies(df):
    """
    Detect anomalous consumption patterns using Isolation Forest
    """
    # Prepare data for anomaly detection
    features = ['consumption', 'temperature', 'hour', 'day_of_week', 'is_weekend']
    
    # Organize by region
    regions = df['region'].unique()
    all_anomalies = []
    
    for region in regions:
        region_data = df[df['region'] == region].copy()
        
        # Scale the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(region_data[features])
        
        # Run Isolation Forest
        isolation_forest = IsolationForest(
            contamination=0.05,  # Assume 5% anomalies
            random_state=42
        )
        
        # Fit and predict
        region_data['anomaly'] = isolation_forest.fit_predict(scaled_features)
        
        # -1 for anomalies, 1 for regular data
        region_data['anomaly'] = region_data['anomaly'].map({1: 0, -1: 1})
        
        # Get anomalies
        anomalies = region_data[region_data['anomaly'] == 1]
        all_anomalies.append(anomalies)
        
        print(f"\nRegion {region}: Detected {len(anomalies)} anomalies out of {len(region_data)} records ({len(anomalies)/len(region_data)*100:.2f}%)")
        
        # Analyze anomalies
        high_temp_anomalies = anomalies[anomalies['temperature'] > 85].shape[0]
        low_temp_anomalies = anomalies[anomalies['temperature'] < 30].shape[0]
        weekend_anomalies = anomalies[anomalies['is_weekend'] == 1].shape[0]
        
        print(f"  - High temperature anomalies: {high_temp_anomalies} ({high_temp_anomalies/len(anomalies)*100:.2f}%)")
        print(f"  - Low temperature anomalies: {low_temp_anomalies} ({low_temp_anomalies/len(anomalies)*100:.2f}%)")
        print(f"  - Weekend anomalies: {weekend_anomalies} ({weekend_anomalies/len(anomalies)*100:.2f}%)")
        
        # Plot consumption with anomalies highlighted
        plt.figure(figsize=(15, 6))
        plt.scatter(
            region_data[region_data['anomaly'] == 0].index, 
            region_data[region_data['anomaly'] == 0]['consumption'],
            c='blue', label='Normal', alpha=0.5
        )
        
        plt.scatter(
            region_data[region_data['anomaly'] == 1].index, 
            region_data[region_data['anomaly'] == 1]['consumption'],
            c='red', label='Anomaly', alpha=0.7
        )
        
        plt.title(f'Consumption Anomalies for {region}')
        plt.ylabel('Consumption')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{region}_anomalies.png')
    
    # Combine all anomalies
    all_anomalies_df = pd.concat(all_anomalies)
    
    # Analyze overall anomaly patterns
    temp_related = (all_anomalies_df['temperature'] > 85) | (all_anomalies_df['temperature'] < 30)
    percent_temp_related = temp_related.sum() / len(all_anomalies_df) * 100
    
    print(f"\nOverall Anomaly Analysis:")
    print(f"Total anomalies detected: {len(all_anomalies_df)}")
    print(f"Temperature-related anomalies: {temp_related.sum()} ({percent_temp_related:.2f}%)")
    
    return {
        'total_anomalies': len(all_anomalies_df),
        'percent_temp_related': percent_temp_related,
        'anomaly_data': all_anomalies_df
    }

# 7. Renewable Energy Optimization
def renewable_optimization(df):
    """
    Analyze and optimize renewable energy mix for business impact
    """
    # Extract regions
    regions = df['region'].unique()
    results = {}
    
    for region in regions:
        region_data = df[df['region'] == region].copy()
        
        # Create daily aggregates for better analysis
        daily_data = region_data.groupby(region_data['timestamp'].dt.date).agg({
            'consumption': 'mean',
            'renewable_percentage': 'mean',
            'business_activity': 'mean',
            'temperature': 'mean'
        }).reset_index()
        
        # Find correlation between renewable % and business activity
        renewable_business_corr = daily_data['renewable_percentage'].corr(daily_data['business_activity'])
        
        # Build model to predict business activity
        X = daily_data[['renewable_percentage', 'consumption', 'temperature']]
        y = daily_data['business_activity']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        
        # Evaluate
        preds = gb_model.predict(X_test)
        r2 = r2_score(y_test, preds)
        
        # Simulate increasing renewable percentage
        scenarios = []
        current_renewable = region_data['renewable_percentage'].mean()
        
        for increase in [5, 10, 15, 20]:
            new_renewable = min(current_renewable + increase, 100)
            
            # Create a copy of test data and modify renewable %
            X_modified = X_test.copy()
            X_modified['renewable_percentage'] = new_renewable
            
            # Predict business activity
            new_preds = gb_model.predict(X_modified)
            
            # Calculate percent improvement
            baseline = gb_model.predict(X_test).mean()
            improvement = (new_preds.mean() - baseline) / baseline * 100
            
            scenarios.append({
                'increase': increase,
                'new_renewable': new_renewable,
                'baseline_business': baseline,
                'projected_business': new_preds.mean(),
                'percent_improvement': improvement
            })
        
        results[region] = {
            'renewable_business_corr': renewable_business_corr,
            'model_r2': r2,
            'scenarios': scenarios,
            'current_renewable': current_renewable
        }
        
        # Calculate optimal renewable percentage
        best_scenario = max(scenarios, key=lambda x: x['percent_improvement'])
        
        print(f"\nRenewable Energy Optimization for {region}:")
        print(f"Current renewable percentage: {current_renewable:.2f}%")
        print(f"Correlation with business activity: {renewable_business_corr:.4f}")
        print(f"Model R² score: {r2:.4f}")
        print(f"Best scenario: +{best_scenario['increase']}% renewable energy")
        print(f"Projected business impact: +{best_scenario['percent_improvement']:.2f}%")
    
    # Find region with highest potential impact
    highest_impact_region = max(results.items(), 
                               key=lambda x: max(s['percent_improvement'] for s in x[1]['scenarios']))
    
    print(f"\nRegion with highest potential impact: {highest_impact_region[0]}")
    print(f"Potential business improvement: {max(s['percent_improvement'] for s in highest_impact_region[1]['scenarios']):.2f}%")
    
    return results

# 8. Main Analysis Pipeline
def run_full_analysis():
    """
    Execute the complete analysis pipeline
    """
    # Generate synthetic data since we don't have real data
    print("Generating synthetic energy consumption data...")
    df = generate_synthetic_data(num_regions=5, days=60)
    print(f"Generated dataset with {len(df)} records across {df['region'].nunique()} regions.")
    
    # Exploratory Data Analysis
    print("\n===== Exploratory Data Analysis =====")
    corr_matrix = perform_eda(df)
    
    # Time Series Forecasting
    print("\n===== Time Series Forecasting =====")
    forecast_results = {}
    for region in df['region'].unique():
        forecast_results[region] = time_series_forecast(df, region)
    
    # Consumption Prediction Model
    print("\n===== Consumption Prediction Model =====")
    consumption_model_results = build_consumption_model(df)
    
    # Health Impact Analysis
    print("\n===== Health Impact Analysis =====")
    health_impact_results = health_impact_model(df)
    
    # Anomaly Detection
    print("\n===== Anomaly Detection =====")
    anomaly_results = detect_anomalies(df)
    
    # Renewable Energy Optimization
    print("\n===== Renewable Energy Optimization =====")
    renewable_results = renewable_optimization(df)
    
    # Compile final results
    final_results = {
        'forecast_results': forecast_results,
        'consumption_model': consumption_model_results,
        'health_impact': health_impact_results,
        'anomalies': anomaly_results,
        'renewable_optimization': renewable_results
    }
    
    print("\n===== Analysis Complete =====")
    print("Key findings:")
    print(f"1. Time series models can predict consumption with RMSE ranging from " +
          f"{min(r['prophet_rmse'] for r in forecast_results.values()):.2f} to " +
          f"{max(r['prophet_rmse'] for r in forecast_results.values()):.2f}")
    
    print(f"2. Top factors affecting consumption: " +
          ", ".join(consumption_model_results['feature_importance']['Feature'].head(3).tolist()))
    
    print(f"3. A 10% reduction in peak consumption could reduce hospital admissions by " +
          f"{health_impact_results['percent_reduction']:.2f}%")
    
    print(f"4. {anomaly_results['percent_temp_related']:.1f}% of consumption anomalies are temperature-related")
    
    print(f"5. Optimizing renewable energy could improve business metrics by up to " +
          f"{max(max(s['percent_improvement'] for s in region['scenarios']) for region in renewable_results.values()):.2f}%")
    
    return final_results

# Run the analysis when executed directly
if __name__ == "__main__":
    run_full_analysis()