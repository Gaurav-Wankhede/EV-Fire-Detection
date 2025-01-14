import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
import os

# Get API URL from environment variable or use default
API_URL = os.getenv('API_URL', 'http://localhost:8000')

# Set page config
st.set_page_config(
    page_title="EV Battery Fire Risk Analysis",
    page_icon="",
    layout="wide"
)

# Title and description
st.title("")
st.markdown("""
This application analyzes the fire risk in EV batteries based on key parameters:
- Temperature
- Voltage
- State of Charge (SoC)
""")

# Input parameters
st.sidebar.header("Battery Parameters")

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=-20.0,
    max_value=100.0,
    value=25.0,
    step=0.1,
    help="Battery temperature in Celsius"
)

voltage = st.sidebar.slider(
    "Voltage (V)",
    min_value=0.0,
    max_value=5.0,
    value=3.7,
    step=0.01,
    help="Battery voltage"
)

soc = st.sidebar.slider(
    "State of Charge (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=0.1,
    help="Battery State of Charge"
)

# Create input data
data = {
    "temperature": temperature,
    "voltage": voltage,
    "soc": soc
}

try:
    # Make prediction using environment variable
    response = requests.post(f"{API_URL}/predict", json=data)
    response.raise_for_status()  # Raise an error for bad responses
    result = response.json()

    # Display results in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Create gauge chart for final risk score
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result['fire_risk_score'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fire Risk Score", 'font': {'size': 24}},
            delta={'reference': 60, 'increasing': {'color': "red"}},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "darkblue",
                    'tickmode': 'linear',
                    'tick0': 0,
                    'dtick': 5
                },
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': 'green'},
                    {'range': [20, 40], 'color': 'lightgreen'},
                    {'range': [40, 60], 'color': 'yellow'},
                    {'range': [60, 80], 'color': 'orange'},
                    {'range': [80, 100], 'color': 'red'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show risk level explanation and recommendations
        risk_score = result['fire_risk_score'] * 100
        st.markdown("### Risk Level Analysis and Safety Recommendations")
        
        if risk_score >= 80:
            st.error(f"""
            ### CRITICAL RISK LEVEL: {risk_score:.1f}%
            
            EMERGENCY ACTIONS REQUIRED:
            1. Immediately stop the vehicle in a safe location
            2. Evacuate all passengers
            3. Contact emergency services
            4. Maintain safe distance (at least 50 meters)
            5. Alert nearby people
            
            DO NOT:
            - Attempt to charge the vehicle
            - Try to move the vehicle
            - Open the battery compartment
            - Use water for fire suppression
            - Touch any high-voltage components
            
            FOR TECHNICIANS:
            - Use Class D fire extinguisher only
            - Monitor thermal camera readings
            - Isolate high-voltage system
            - Document all parameters
            - Use appropriate PPE
            """)
            
        elif risk_score >= 60:
            st.warning(f"""
            ### HIGH RISK LEVEL: {risk_score:.1f}%
            
            REQUIRED ACTIONS:
            1. Park vehicle in open area
            2. Avoid charging
            3. Schedule immediate inspection
            4. Monitor battery temperature
            5. Keep fire extinguisher nearby
            
            DO NOT:
            - Park in enclosed spaces
            - Fast charge the battery
            - Ignore warning signals
            - Continue normal operation
            - Override safety systems
            
            MAINTENANCE REQUIRED:
            - Check cooling system
            - Inspect battery connections
            - Verify sensor readings
            - Test safety systems
            - Log all parameters
            """)
            
        elif risk_score >= 40:
            st.warning(f"""
            ### MODERATE RISK LEVEL: {risk_score:.1f}%
            
            RECOMMENDED ACTIONS:
            1. Reduce charging frequency
            2. Monitor battery parameters
            3. Plan maintenance check
            4. Avoid extreme conditions
            5. Check warning systems
            
            AVOID:
            - Rapid acceleration/deceleration
            - Extended parking in hot areas
            - Frequent fast charging
            - Ignoring minor warnings
            - Deep discharge cycles
            
            PREVENTIVE MEASURES:
            - Schedule diagnostic scan
            - Clean battery contacts
            - Check ventilation system
            - Update battery management system
            - Record performance data
            """)
            
        elif risk_score >= 20:
            st.info(f"""
            ### LOW RISK LEVEL: {risk_score:.1f}%
            
            GOOD PRACTICES:
            1. Follow charging schedule
            2. Regular visual inspections
            3. Maintain service records
            4. Monitor performance
            5. Keep battery level optimal
            
            AVOID:
            - Exposure to extreme weather
            - Unauthorized modifications
            - Using non-standard chargers
            - Skipping maintenance
            - Overloading vehicle
            
            MAINTENANCE TIPS:
            - Check battery health monthly
            - Clean charging port
            - Verify cooling function
            - Inspect cable conditions
            - Update software regularly
            """)
            
        else:
            st.success(f"""
            ### MINIMAL RISK LEVEL: {risk_score:.1f}%
            
            BEST PRACTICES:
            1. Follow manufacturer guidelines
            2. Regular maintenance schedule
            3. Use recommended chargers
            4. Keep records updated
            5. Monitor system alerts
            
            STILL AVOID:
            - Unauthorized repairs
            - Non-standard accessories
            - Extreme operating conditions
            - Ignoring minor issues
            - DIY modifications
            
            PREVENTIVE CARE:
            - Annual system check
            - Battery health monitoring
            - Software updates
            - Charging system inspection
            - Performance logging
            """)
        
        # Show severity level
        st.markdown(f"### Severity Level: **{result['severity']}**")
        
        # Show binary prediction
        if result['binary_prediction'] == 1:
            st.error("")
        else:
            st.success("")
    
    with col2:
        # Display critical conditions if present
        if result.get('critical_conditions'):
            st.markdown("### Critical Conditions Detected")
            for condition in result['critical_conditions']:
                if "EXTREME" in condition:
                    st.error(f" {condition}")
                elif "CRITICAL" in condition:
                    st.error(f" {condition}")
                else:
                    st.warning(f" {condition}")
        
        # Parameter status
        st.markdown("### Parameter Status")
        
        # Temperature status
        temp_status = (
            " CRITICAL" if temperature > 65 else
            " HIGH RISK" if temperature > 55 else
            " WARNING" if temperature > 45 else
            " NORMAL" if 15 <= temperature <= 45 else
            " LOW"
        )
        st.markdown(f"""
            #### Temperature Status: {temp_status}
            Current: **{temperature}°C**
            
            Thresholds:
            - Critical: >65°C
            - High Risk: >55°C
            - Warning: >45°C
            - Normal: 15-45°C
            - Low: <15°C
        """)
        
        # Voltage status
        voltage_status = (
            " CRITICAL HIGH" if voltage > 4.25 else
            " HIGH" if voltage > 4.2 else
            " NORMAL" if 3.0 <= voltage <= 4.2 else
            " LOW" if voltage > 2.5 else
            " CRITICAL LOW"
        )
        st.markdown(f"""
            #### Voltage Status: {voltage_status}
            Current: **{voltage}V**
            
            Thresholds:
            - Critical High: >4.25V
            - High: >4.2V
            - Normal: 3.0-4.2V
            - Low: <3.0V
            - Critical Low: <2.5V
        """)
        
        # SoC status
        soc_status = (
            " CRITICAL HIGH" if soc > 95 else
            " HIGH" if soc > 90 else
            " NORMAL" if 10 <= soc <= 90 else
            " LOW" if soc > 5 else
            " CRITICAL LOW"
        )
        st.markdown(f"""
            #### State of Charge Status: {soc_status}
            Current: **{soc}%**
            
            Thresholds:
            - Critical High: >95%
            - High: >90%
            - Normal: 10-90%
            - Low: <10%
            - Critical Low: <5%
        """)

except requests.exceptions.ConnectionError:
    st.error("""
    ### Connection Error
    
    The FastAPI server is not running. Please start the server using:
    ```
    uvicorn api:app --reload
    ```
    """)
except Exception as e:
    st.error(f"""
    ### Error
    
    An error occurred: {str(e)}
    """)
