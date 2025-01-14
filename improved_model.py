import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load and preprocess data
def load_data(file_path):
    df = pd.read_excel(file_path)
    
    # Handle negative SOC values
    df['SoC'] = df['SoC'].apply(lambda x: x if x >= 0 else np.nan)
    df['SoC'].fillna(df['SoC'].median(), inplace=True)
    
    # Convert temperature to Celsius and clip values
    df['Temperature'] = df['Temperature'] - 273.15
    df['Temperature'] = np.clip(df['Temperature'], -20, 100)
    df['Voltage'] = np.clip(df['Voltage'], 0, 5)
    df['SoC'] = np.clip(df['SoC'], 0, 100)
    
    return df

# Create enhanced features
def create_features(df):
    # Original features
    features = df[['SoC', 'Temperature', 'Voltage']].copy()
    
    # Add interaction terms
    features['Temp_Voltage'] = df['Temperature'] * df['Voltage']
    features['Temp_SoC'] = df['Temperature'] * df['SoC']
    features['Voltage_SoC'] = df['Voltage'] * df['SoC']
    
    # Add polynomial terms
    features['Temp_Squared'] = df['Temperature'] ** 2
    features['Voltage_Squared'] = df['Voltage'] ** 2
    features['SoC_Squared'] = df['SoC'] ** 2
    
    # Add threshold-based features
    features['High_Temp'] = (df['Temperature'] > 45).astype(int)
    features['Low_Voltage'] = (df['Voltage'] < 2.5).astype(int)
    features['High_Voltage'] = (df['Voltage'] > 4.2).astype(int)
    features['Low_SoC'] = (df['SoC'] < 20).astype(int)
    features['High_SoC'] = (df['SoC'] > 90).astype(int)
    
    return features

# Calculate improved fire risk score
def calculate_fire_risk_score(df):
    risk_score = np.zeros(len(df))
    
    # Temperature risk (exponential increase after threshold)
    temp_risk = np.exp((df['Temperature'] - 45) / 10) / np.exp(5.5)  # Normalized
    risk_score += np.clip(temp_risk, 0, 0.4)
    
    # Voltage risk (both low and high voltage are risky)
    voltage_low_risk = np.exp((2.5 - df['Voltage']) * 2) / np.exp(5)
    voltage_high_risk = np.exp((df['Voltage'] - 4.2) * 2) / np.exp(5)
    risk_score += np.clip(voltage_low_risk + voltage_high_risk, 0, 0.3)
    
    # SoC risk (both very low and very high are risky)
    soc_low_risk = np.exp((20 - df['SoC']) / 20) / np.exp(1)
    soc_high_risk = np.exp((df['SoC'] - 90) / 10) / np.exp(1)
    risk_score += np.clip(soc_low_risk + soc_high_risk, 0, 0.3)
    
    # Additional risk for labeled faults
    risk_score[df['Label'].isin([1, 2])] += 0.4
    
    # Normalize to [0, 1]
    return np.clip(risk_score, 0, 1)

def train_model():
    # Load data
    df = load_data('Multiple Classification - EV Battery Faults Dataset.xlsx')
    
    # Create features and target
    X = create_features(df)
    y = calculate_fire_risk_score(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }
    
    with open('model/improved_fire_risk_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
