# Flask Backend API for F1 Strategy Analysis
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global variables to store models and data
rf_stint_model = None
rf_pitstops_model = None
le_driver = None
le_gp = None
le_compound = None
le_strategy = None
circuit_df = None
merged_df = None

def load_models_and_data():
    """Load all trained models and data"""
    global rf_stint_model, rf_pitstops_model, le_driver, le_gp, le_compound, le_strategy, circuit_df, merged_df
    
    try:
        # Load models
        with open('rf_stint_model.pkl', 'rb') as f:
            rf_stint_model = pickle.load(f)
        
        with open('rf_pitstops_model.pkl', 'rb') as f:
            rf_pitstops_model = pickle.load(f)
        
        # Load encoders
        with open('le_driver.pkl', 'rb') as f:
            le_driver = pickle.load(f)
        
        with open('le_gp.pkl', 'rb') as f:
            le_gp = pickle.load(f)
        
        with open('le_compound.pkl', 'rb') as f:
            le_compound = pickle.load(f)
        
        with open('le_strategy.pkl', 'rb') as f:
            le_strategy = pickle.load(f)
        
        # Load data
        circuit_df = pd.read_csv('CircuitInfo.csv')
        strategy_df = pd.read_csv('Strategyfull.csv')
        strategy_df_clean = strategy_df.dropna(subset=['Driver', 'Strategy', 'Stint', 'Compound', 'StintLength'])
        merged_df = strategy_df_clean.merge(circuit_df, on='GP', how='left')
        
        # Fill missing values
        circuit_cols = ['Traction', 'Braking', 'TrackEvo']
        for col in circuit_cols:
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())
        
        print("✅ Models and data loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def predict_strategy(year, driver, gp, compound, stint_number, weather_factor=1.0):
    """
    Predict optimal strategy based on input parameters
    """
    try:
        # Get circuit information
        circuit_info = circuit_df[circuit_df['GP'] == gp]
        if circuit_info.empty:
            return {'error': f'Circuit {gp} not found in database'}
        
        circuit_data = circuit_info.iloc[0]
        
        # Encode categorical variables
        driver_encoded = le_driver.transform([driver])[0] if driver in le_driver.classes_ else 0
        gp_encoded = le_gp.transform([gp])[0] if gp in le_gp.classes_ else 0
        compound_encoded = le_compound.transform([compound])[0] if compound in le_compound.classes_ else 0
        
        # Create feature vector
        features = np.array([[
            year, driver_encoded, gp_encoded, compound_encoded, stint_number,
            circuit_data['Length'], circuit_data['Abrasion'], circuit_data['Traction'],
            circuit_data['Braking'], circuit_data['TrackEvo'], circuit_data['Grip'],
            circuit_data['Lateral'], circuit_data['Downforce'], circuit_data['TyreStress']
        ]])
        
        # Make predictions
        stint_pred = rf_stint_model.predict(features)[0] * weather_factor
        pitstops_pred = max(1, round(rf_pitstops_model.predict(features)[0]))
        
        # Get historical data for this circuit
        historical_data = merged_df[merged_df['GP'] == gp]
        
        # Generate strategy recommendations
        strategy_recommendations = generate_strategy_recommendations(
            gp, stint_pred, pitstops_pred, compound, historical_data
        )
        
        return {
            'success': True,
            'predictions': {
                'optimal_stint_length': round(stint_pred, 1),
                'optimal_pitstops': pitstops_pred,
                'total_race_distance': round(stint_pred * pitstops_pred, 1)
            },
            'strategy': strategy_recommendations,
            'circuit_info': {
                'name': gp,
                'length': round(circuit_data['Length'], 3),
                'abrasion_level': int(circuit_data['Abrasion']),
                'grip_level': int(circuit_data['Grip']),
                'tyre_stress': int(circuit_data['TyreStress'])
            },
            'confidence': {
                'stint_prediction': 'High' if rf_stint_model.score else 'Medium',
                'pitstop_prediction': 'High'
            }
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def generate_strategy_recommendations(circuit, stint_length, pitstops, compound, historical_data):
    """
    Generate comprehensive strategy recommendations
    """
    # Get most successful historical strategies for this circuit
    if not historical_data.empty:
        top_strategies = historical_data['Strategy'].value_counts().head(3)
        best_compounds = historical_data['Compound'].value_counts().head(3)
        avg_stint = round(historical_data['StintLength'].mean(), 1)
    else:
        top_strategies = pd.Series(['MEDIUM-HARD', 'SOFT-MEDIUM', 'HARD-MEDIUM'])
        best_compounds = pd.Series(['MEDIUM', 'HARD', 'SOFT'])
        avg_stint = stint_length
    
    # Generate compound sequence based on pit stops
    if pitstops == 1:
        compound_sequence = ['MEDIUM', 'HARD']
        strategy_type = "One-Stop Strategy"
        risk_level = "Medium"
    elif pitstops == 2:
        compound_sequence = ['SOFT', 'MEDIUM', 'HARD']
        strategy_type = "Two-Stop Strategy"  
        risk_level = "Low"
    else:
        compound_sequence = ['SOFT', 'MEDIUM', 'MEDIUM', 'HARD']
        strategy_type = "Multi-Stop Strategy"
        risk_level = "High"
    
    return {
        'primary_strategy': {
            'type': strategy_type,
            'compounds': compound_sequence,
            'stint_lengths': [round(stint_length)] * len(compound_sequence),
            'pit_windows': generate_pit_windows(stint_length, pitstops),
            'risk_level': risk_level
        },
        'alternative_strategies': [
            {
                'name': 'Conservative',
                'compounds': ['HARD', 'MEDIUM'],
                'description': 'Lower risk, consistent pace'
            },
            {
                'name': 'Aggressive',  
                'compounds': ['SOFT', 'SOFT', 'HARD'],
                'description': 'High risk, maximum attack'
            }
        ],
        'historical_insights': {
            'circuit_avg_stint': avg_stint,
            'most_successful_strategy': top_strategies.index[0] if len(top_strategies) > 0 else 'MEDIUM-HARD',
            'preferred_compounds': list(best_compounds.index[:2])
        }
    }

def generate_pit_windows(stint_length, pitstops):
    """Generate optimal pit stop windows"""
    if pitstops == 1:
        return [round(stint_length * 0.6)]
    elif pitstops == 2:
        return [round(stint_length * 0.4), round(stint_length * 0.7)]
    else:
        windows = []
        for i in range(pitstops):
            windows.append(round(stint_length * (0.25 + i * 0.25)))
        return windows

@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/circuits', methods=['GET'])
def get_circuits():
    """Get all available circuits"""
    try:
        circuits = sorted(circuit_df['GP'].unique())
        return jsonify({'circuits': circuits})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/drivers', methods=['GET'])
def get_drivers():
    """Get all available drivers"""
    try:
        drivers = sorted([d for d in le_driver.classes_ if pd.notna(d)])
        return jsonify({'drivers': drivers})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compounds', methods=['GET'])
def get_compounds():
    """Get all available tire compounds"""
    try:
        compounds = sorted([c for c in le_compound.classes_ if c not in ['UNKNOWN'] and pd.notna(c)])
        return jsonify({'compounds': compounds})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_strategy():
    """Main API endpoint for strategy analysis"""
    try:
        data = request.json
        
        # Extract parameters
        circuit = data.get('circuit')
        driver = data.get('driver') 
        compound = data.get('compound', 'MEDIUM')
        grid_position = int(data.get('grid_position', 10))
        rain_probability = float(data.get('rain_probability', 0))
        rain_intensity = data.get('rain_intensity', 'None')
        
        # Calculate weather factor
        weather_factor = 1.0
        if rain_probability > 50:
            if rain_intensity == 'Heavy':
                weather_factor = 1.4
                compound = 'WET'
            elif rain_intensity == 'Medium':
                weather_factor = 1.2
                compound = 'INTERMEDIATE'
            else:
                weather_factor = 1.1
        
        # Predict strategy
        result = predict_strategy(
            year=2024,
            driver=driver,
            gp=circuit,
            compound=compound,
            stint_number=1,
            weather_factor=weather_factor
        )
        
        if result.get('success'):
            # Add weather and grid position adjustments
            result['weather_impact'] = {
                'rain_probability': rain_probability,
                'rain_intensity': rain_intensity,
                'recommended_compound': compound,
                'strategy_adjustment': get_weather_adjustment(rain_probability, rain_intensity)
            }
            
            result['grid_position_impact'] = {
                'starting_position': grid_position,
                'strategy_adjustment': get_grid_position_adjustment(grid_position)
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def get_weather_adjustment(rain_prob, intensity):
    """Get weather-based strategy adjustments"""
    if rain_prob < 20:
        return "Standard dry strategy recommended"
    elif rain_prob < 50:
        return "Have intermediates ready, extended stint possible"
    elif rain_prob < 80:
        return f"High rain probability - {intensity.lower()} rain expected, flexible pit strategy needed"
    else:
        return "Wet race likely - safety car periods expected, opportunistic strategy recommended"

def get_grid_position_adjustment(grid_pos):
    """Get grid position-based strategy adjustments"""
    if grid_pos <= 3:
        return "Front-row start: Aggressive strategy possible, cover competitors"
    elif grid_pos <= 10:
        return "Midfield start: Alternative strategy recommended for track position"
    else:
        return "Back of grid: High-risk undercut strategy, longer first stint"

if __name__ == '__main__':
    print("🏎️  Starting F1 Strategy Analysis Server...")
    
    # Load models and data
    if load_models_and_data():
        print("🚀 Server ready! Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load models. Please ensure all .pkl files are present.")