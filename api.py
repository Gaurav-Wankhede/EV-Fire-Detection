import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Load the regression model
try:
    with open('model/fire_risk_model.pkl', 'rb') as file:
        model = pickle.load(file)  # Load model directly
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

app = FastAPI(title="Fire Risk Prediction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For Cloud Run, we'll allow all origins initially
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    temperature: float
    voltage: float
    soc: float  # State of Charge

class PredictionOutput(BaseModel):
    fire_risk_score: float  # Continuous value between 0 and 1
    severity: str  # Risk severity level
    binary_prediction: int  # 0 or 1 based on 0.6 threshold

@app.get("/")
def read_root():
    return {"message": "Fire Risk Prediction API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def get_severity_level(score: float) -> str:
    """Determine severity level based on risk score thresholds"""
    if score <= 0.2:
        return "Very Low"
    elif score <= 0.4:
        return "Low"
    elif score <= 0.6:
        return "Medium"
    elif score <= 0.8:
        return "High"
    else:
        return "Very High"

def create_features(input_data):
    """Create features for model prediction"""
    # Base features
    soc = input_data.soc
    temp = input_data.temperature
    voltage = input_data.voltage
    
    # Create all 14 features
    features = np.array([[
        soc,  # SoC
        temp,  # Temperature
        voltage,  # Voltage
        temp * voltage,  # Temp_Voltage
        temp * soc,  # Temp_SoC
        voltage * soc,  # Voltage_SoC
        temp ** 2,  # Temp_Squared
        voltage ** 2,  # Voltage_Squared
        soc ** 2,  # SoC_Squared
        1 if temp > 45 else 0,  # High_Temp
        1 if voltage < 2.5 else 0,  # Low_Voltage
        1 if voltage > 4.2 else 0,  # High_Voltage
        1 if soc < 20 else 0,  # Low_SoC
        1 if soc > 90 else 0  # High_SoC
    ]])
    
    return features

def check_critical_conditions(temp, voltage, soc):
    """Check for critical conditions based on real-world EV battery specifications"""
    critical_conditions = []
    
    # Temperature critical conditions (based on Li-ion battery safety limits)
    if temp > 65:  # Critical thermal runaway threshold
        critical_conditions.append("EXTREME DANGER: Temperature above thermal runaway threshold (>65°C)")
    elif temp > 55:  # High-risk temperature zone
        critical_conditions.append("CRITICAL: Temperature in high-risk zone (>55°C)")
        if voltage > 4.1 or voltage < 2.8:
            critical_conditions.append("CRITICAL: High temperature with dangerous voltage levels")
        if soc > 80:
            critical_conditions.append("CRITICAL: High temperature with high charge state")
    elif temp > 45:  # Warning temperature zone
        if voltage > 4.15 or voltage < 2.5:
            critical_conditions.append("WARNING: Elevated temperature with concerning voltage levels")
        if soc > 90 or soc < 10:
            critical_conditions.append("WARNING: Elevated temperature with extreme charge state")
    elif temp < -10:  # Cold temperature warnings
        if soc < 30:
            critical_conditions.append("WARNING: Low temperature with low charge - risk of lithium plating")
        if voltage < 3.0:
            critical_conditions.append("WARNING: Low temperature with low voltage - risk of cell damage")
    
    # Voltage critical conditions (based on typical Li-ion cell limits)
    if voltage > 4.25:  # Absolute maximum voltage
        critical_conditions.append("CRITICAL: Voltage exceeds safe maximum (>4.25V)")
    elif voltage > 4.2:  # High voltage warning
        if soc > 90:
            critical_conditions.append("WARNING: High voltage with high charge state")
        if temp > 40:
            critical_conditions.append("WARNING: High voltage with elevated temperature")
    elif voltage < 2.5:  # Low voltage warning
        critical_conditions.append("CRITICAL: Voltage below safe minimum (<2.5V)")
        if temp > 35:
            critical_conditions.append("CRITICAL: Low voltage with elevated temperature")
    
    # State of Charge (SoC) critical conditions
    if soc > 95:  # Very high SoC
        critical_conditions.append("WARNING: Extremely high charge state (>95%)")
        if temp > 35:
            critical_conditions.append("CRITICAL: High charge with elevated temperature")
    elif soc < 5:  # Very low SoC
        critical_conditions.append("WARNING: Extremely low charge state (<5%)")
        if temp < 0:
            critical_conditions.append("CRITICAL: Low charge with low temperature")
    
    # Combined critical conditions (based on real-world failure scenarios)
    if temp > 45 and voltage > 4.2 and soc > 90:
        critical_conditions.append("CRITICAL: Multiple high-risk factors - High temperature, voltage, and charge state")
    if temp > 50 and voltage < 3.0:
        critical_conditions.append("CRITICAL: Thermal event risk - High temperature with low voltage")
    if temp > 40 and voltage > 4.15 and soc > 85:
        critical_conditions.append("WARNING: Approaching thermal runaway conditions")
    
    # Rate of change warnings (if historical data available)
    # Note: This would require implementing temperature rate monitoring
    # if temp_rate > 5:  # °C per minute
    #     critical_conditions.append("CRITICAL: Rapid temperature increase detected")
    
    return critical_conditions

def calculate_risk_score(features):
    """Calculate fire risk score based on real-world battery safety parameters"""
    # Extract base features
    soc = features[0][0]
    temp = features[0][1]
    voltage = features[0][2]
    
    # Initialize risk score components
    temp_risk = 0.0
    voltage_risk = 0.0
    soc_risk = 0.0
    
    # Temperature risk (40% weight)
    if temp > 65:
        temp_risk = 1.0
    elif temp > 55:
        temp_risk = 0.8 + (temp - 55) * 0.02
    elif temp > 45:
        temp_risk = 0.5 + (temp - 45) * 0.03
    elif temp > 35:
        temp_risk = 0.3 + (temp - 35) * 0.02
    elif temp < -10:
        temp_risk = 0.3 + abs(temp + 10) * 0.02
    else:
        temp_risk = max(0.1, (temp - 25) * 0.01)
    
    # Voltage risk (35% weight)
    if voltage > 4.25:
        voltage_risk = 1.0
    elif voltage > 4.2:
        voltage_risk = 0.8 + (voltage - 4.2) * 4
    elif voltage < 2.5:
        voltage_risk = 0.8 + (2.5 - voltage) * 0.4
    elif voltage > 4.1:
        voltage_risk = 0.4 + (voltage - 4.1) * 4
    elif voltage < 3.0:
        voltage_risk = 0.4 + (3.0 - voltage) * 0.4
    else:
        voltage_risk = abs(voltage - 3.7) * 0.2
    
    # SoC risk (25% weight)
    if soc > 95:
        soc_risk = 0.8 + (soc - 95) * 0.04
    elif soc < 5:
        soc_risk = 0.8 + (5 - soc) * 0.04
    elif soc > 90:
        soc_risk = 0.4 + (soc - 90) * 0.08
    elif soc < 10:
        soc_risk = 0.4 + (10 - soc) * 0.08
    else:
        soc_risk = abs(soc - 50) * 0.004
    
    # Calculate weighted risk score
    risk_score = (
        0.40 * temp_risk +
        0.35 * voltage_risk +
        0.25 * soc_risk
    )
    
    # Additional risk factors for combined conditions
    if temp > 45 and voltage > 4.2:
        risk_score = min(1.0, risk_score * 1.3)
    if temp > 50 and soc > 90:
        risk_score = min(1.0, risk_score * 1.4)
    if voltage < 2.8 and temp > 40:
        risk_score = min(1.0, risk_score * 1.25)
    
    return np.clip(risk_score, 0, 1)

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Input validation
        if not (0 <= input_data.soc <= 100):
            raise ValueError("State of Charge must be between 0 and 100%")
        if not (0 <= input_data.voltage <= 5):
            raise ValueError("Voltage must be between 0 and 5V")
        if not (-20 <= input_data.temperature <= 100):
            raise ValueError("Temperature must be between -20 and 100°C")
        
        # Create features
        X = create_features(input_data)
        
        # Get raw model prediction
        raw_model_prediction = float(model.predict(X)[0])
        
        # Get feature-based risk score
        feature_risk = calculate_risk_score(X)
        
        # Check for critical conditions
        critical_conditions = check_critical_conditions(
            input_data.temperature,
            input_data.voltage,
            input_data.soc
        )
        
        # Calculate final risk score with weighted components
        # Model prediction gets higher weight in normal conditions
        if raw_model_prediction > 0.8 or feature_risk > 0.8:
            # If either score is very high, take the maximum
            final_prediction = max(raw_model_prediction, feature_risk)
        elif critical_conditions:
            # If critical conditions exist, bias towards feature-based risk
            final_prediction = (0.4 * raw_model_prediction) + (0.6 * feature_risk)
        else:
            # Normal conditions: balanced weight
            final_prediction = (0.5 * raw_model_prediction) + (0.5 * feature_risk)
        
        # Ensure prediction is between 0 and 1
        final_prediction = np.clip(final_prediction, 0, 1)
        
        # Get severity level
        severity = get_severity_level(final_prediction)
        
        # Get binary prediction using 0.6 threshold
        binary_prediction = 1 if final_prediction >= 0.6 else 0
        
        # Create detailed response
        response_dict = {
            'fire_risk_score': final_prediction,
            'raw_model_prediction': raw_model_prediction,
            'feature_based_risk': feature_risk,
            'severity': severity,
            'binary_prediction': binary_prediction,
            'critical_conditions': critical_conditions
        }
        
        # Log prediction details
        print(f"""
Detailed Prediction Analysis:
---------------------------
Input Parameters:
- Temperature: {input_data.temperature}°C
- Voltage: {input_data.voltage}V
- State of Charge: {input_data.soc}%

Risk Scores:
- Raw Model Prediction: {raw_model_prediction:.3f} ({raw_model_prediction*100:.1f}%)
- Feature-Based Risk: {feature_risk:.3f} ({feature_risk*100:.1f}%)
- Final Risk Score: {final_prediction:.3f} ({final_prediction*100:.1f}%)

Classification:
- Severity Level: {severity}
- Binary Risk: {"High Risk" if binary_prediction == 1 else "Low Risk"}

Critical Conditions: {', '.join(critical_conditions) if critical_conditions else "None"}
        """)
        
        return response_dict
    except ValueError as ve:
        # Handle validation errors
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        # Log the error for debugging
        print(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while making the prediction"
        )
