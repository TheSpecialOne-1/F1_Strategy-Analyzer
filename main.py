# Fixed F1 Strategy Analysis - ML Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load the data
strategy_df = pd.read_csv('Strategyfull.csv')  # Use relative path
circuit_df = pd.read_csv('CircuitInfo.csv')    # Use relative path

print("Strategy data shape:", strategy_df.shape)
print("\nStrategy data columns:", strategy_df.columns.tolist())
print("\nStrategy data sample:")
print(strategy_df.head())

print("\n\nCircuit data shape:", circuit_df.shape)
print("\nCircuit data columns:", circuit_df.columns.tolist())
print("\nCircuit data sample:")
print(circuit_df.head())

# Check unique values for categorical columns - FIX: Handle NaN values properly
print("\n\nUnique years:", sorted(strategy_df['Year'].unique()))
print("Unique GPs:", sorted([gp for gp in strategy_df['GP'].unique() if pd.notna(gp)]))
print("Unique drivers:", sorted([driver for driver in strategy_df['Driver'].unique() if pd.notna(driver)]))
print("Unique compounds:", sorted([compound for compound in strategy_df['Compound'].unique() if pd.notna(compound)]))

# Let's check for null values
print("\n\nStrategy data null values:")
print(strategy_df.isnull().sum())

print("\n\nCircuit data null values:")
print(circuit_df.isnull().sum())

# Clean the data and prepare for ML
# Remove rows with missing critical data
strategy_df_clean = strategy_df.dropna(subset=['Driver', 'Strategy', 'Stint', 'Compound', 'StintLength'])

print(f"Original rows: {len(strategy_df)}")
print(f"After cleaning: {len(strategy_df_clean)}")

# Merge with circuit info
merged_df = strategy_df_clean.merge(circuit_df, on='GP', how='left')
print(f"After merging with circuit data: {len(merged_df)}")

# Fill missing circuit characteristics with median values
circuit_cols = ['Traction', 'Braking', 'TrackEvo']
for col in circuit_cols:
    merged_df[col] = merged_df[col].fillna(merged_df[col].median())

# Let's look at the data we have for analysis
print("\n\nFinal dataset columns:")
print(merged_df.columns.tolist())
print("\n\nDataset shape:", merged_df.shape)

# Check for the most successful drivers by average stint length
print("\n\nTop drivers by average stint length:")
driver_performance = merged_df.groupby('Driver').agg({
    'StintLength': 'mean',
    'PitStops': 'mean',
    'Year': 'count'
}).round(2)
driver_performance.columns = ['Avg_StintLength', 'Avg_PitStops', 'Races']
driver_performance = driver_performance[driver_performance['Races'] >= 10]
print(driver_performance.sort_values('Avg_StintLength', ascending=False).head(10))

# Check strategy distribution
print("\n\nMost common strategies:")
print(merged_df['Strategy'].value_counts().head(10))

# Prepare ML features and target variables
# Copy the dataframe for ML processing
ml_df = merged_df.copy()

# Encode categorical variables
le_driver = LabelEncoder()
le_gp = LabelEncoder()
le_compound = LabelEncoder()
le_strategy = LabelEncoder()

ml_df['Driver_encoded'] = le_driver.fit_transform(ml_df['Driver'])
ml_df['GP_encoded'] = le_gp.fit_transform(ml_df['GP'])
ml_df['Compound_encoded'] = le_compound.fit_transform(ml_df['Compound'])
ml_df['Strategy_encoded'] = le_strategy.fit_transform(ml_df['Strategy'])

# Select features for the ML model
feature_cols = ['Year', 'Driver_encoded', 'GP_encoded', 'Compound_encoded',
                'StintNumber', 'Length', 'Abrasion', 'Traction', 'Braking',
                'TrackEvo', 'Grip', 'Lateral', 'Downforce', 'TyreStress']

# Target variables
target_stint_length = 'StintLength'
target_pitstops = 'PitStops'

# Prepare the feature matrix
X = ml_df[feature_cols]
y_stint = ml_df[target_stint_length]
y_pitstops = ml_df[target_pitstops]

print("Features shape:", X.shape)
print("Features:", feature_cols)
print("\nTarget stint length shape:", y_stint.shape)
print("Target pitstops shape:", y_pitstops.shape)

# Check for any remaining NaN values
print("\nNaN values in features:")
print(X.isnull().sum())

print("\nNaN values in targets:")
print(f"Stint length: {y_stint.isnull().sum()}")
print(f"Pit stops: {y_pitstops.isnull().sum()}")

# Split the data
X_train, X_test, y_stint_train, y_stint_test = train_test_split(X, y_stint, test_size=0.2, random_state=42)
_, _, y_pitstops_train, y_pitstops_test = train_test_split(X, y_pitstops, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# Train Random Forest models
rf_stint = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_stint.fit(X_train, y_stint_train)

rf_pitstops = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_pitstops.fit(X_train, y_pitstops_train)

# Make predictions
y_stint_pred = rf_stint.predict(X_test)
y_pitstops_pred = rf_pitstops.predict(X_test)

# Evaluate models
print("\n=== STINT LENGTH MODEL PERFORMANCE ===")
print(f"MAE: {mean_absolute_error(y_stint_test, y_stint_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_stint_test, y_stint_pred)):.2f}")
print(f"R2 Score: {r2_score(y_stint_test, y_stint_pred):.3f}")

print("\n=== PIT STOPS MODEL PERFORMANCE ===")
print(f"MAE: {mean_absolute_error(y_pitstops_test, y_pitstops_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_pitstops_test, y_pitstops_pred)):.2f}")
print(f"R2 Score: {r2_score(y_pitstops_test, y_pitstops_pred):.3f}")

# Feature importance
feature_importance_stint = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_stint.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== FEATURE IMPORTANCE FOR STINT LENGTH ===")
print(feature_importance_stint)

feature_importance_pitstops = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_pitstops.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== FEATURE IMPORTANCE FOR PIT STOPS ===")
print(feature_importance_pitstops)

# Save models and encoders
with open('rf_stint_model.pkl', 'wb') as f:
    pickle.dump(rf_stint, f)

with open('rf_pitstops_model.pkl', 'wb') as f:
    pickle.dump(rf_pitstops, f)

with open('le_driver.pkl', 'wb') as f:
    pickle.dump(le_driver, f)

with open('le_gp.pkl', 'wb') as f:
    pickle.dump(le_gp, f)

with open('le_compound.pkl', 'wb') as f:
    pickle.dump(le_compound, f)

with open('le_strategy.pkl', 'wb') as f:
    pickle.dump(le_strategy, f)

# Create prediction function
def predict_strategy(year, driver, gp, compound, stint_number,
                    length, abrasion, traction, braking, trackevo,
                    grip, lateral, downforce, tyrestress):
    """
    Predict optimal stint length and number of pit stops
    """
    try:
        # Encode inputs
        driver_encoded = le_driver.transform([driver])[0] if driver in le_driver.classes_ else 0
        gp_encoded = le_gp.transform([gp])[0] if gp in le_gp.classes_ else 0
        compound_encoded = le_compound.transform([compound])[0] if compound in le_compound.classes_ else 0
        
        # Create feature vector
        features = np.array([[year, driver_encoded, gp_encoded, compound_encoded, stint_number,
                            length, abrasion, traction, braking, trackevo,
                            grip, lateral, downforce, tyrestress]])
        
        # Make predictions
        stint_pred = rf_stint.predict(features)[0]
        pitstops_pred = rf_pitstops.predict(features)[0]
        
        return {
            'optimal_stint_length': round(stint_pred, 1),
            'optimal_pitstops': round(pitstops_pred),
            'confidence_stint': 'High' if rf_stint.score(X_test, y_stint_test) > 0.5 else 'Medium',
            'confidence_pitstops': 'High' if rf_pitstops.score(X_test, y_pitstops_test) > 0.6 else 'Medium'
        }
    except Exception as e:
        return {'error': str(e)}

# Get unique values for frontend
unique_drivers = sorted([d for d in le_driver.classes_ if pd.notna(d)])
unique_gps = sorted([g for g in le_gp.classes_ if pd.notna(g)])
unique_compounds = sorted([c for c in le_compound.classes_ if c not in ['UNKNOWN'] and pd.notna(c)])

print(f"\nUnique drivers ({len(unique_drivers)}): {unique_drivers[:10]}...")
print(f"Unique GPs ({len(unique_gps)}): {unique_gps[:10]}...")
print(f"Unique compounds: {unique_compounds}")

# Test prediction
test_result = predict_strategy(2024, 'HAM', 'Monaco', 'SOFT', 1,
                              3.337, 1, 5, 2, 5, 1, 1, 5, 1)
print("\nTest prediction for Hamilton at Monaco:")
print(test_result)

# Create team mapping
team_mapping = {
    'HAM': 'Mercedes', 'BOT': 'Mercedes', 'RUS': 'Mercedes',
    'VER': 'Red Bull', 'PER': 'Red Bull', 'GAS': 'Red Bull/AlphaTauri', 'ALB': 'Red Bull/Williams',
    'LEC': 'Ferrari', 'VET': 'Ferrari', 'SAI': 'Ferrari',
    'NOR': 'McLaren', 'RIC': 'McLaren', 'PIA': 'McLaren',
    'ALO': 'Alpine', 'OCO': 'Alpine',
    'STR': 'Aston Martin',
    'MAG': 'Haas', 'GRO': 'Haas', 'MSC': 'Haas', 'HUL': 'Haas',
    'TSU': 'AlphaTauri', 'KVY': 'AlphaTauri',
    'ZHO': 'Alfa Romeo', 'RAI': 'Alfa Romeo', 'GIO': 'Alfa Romeo',
    'LAT': 'Williams', 'DEV': 'Williams', 'FIT': 'Williams'
}

# Analysis function
def analyze_driver_team_strategies():
    analysis = {}
    
    for driver in unique_drivers[:10]:
        if driver in team_mapping:
            team = team_mapping[driver]
        else:
            team = 'Unknown'
            
        driver_data = merged_df[merged_df['Driver'] == driver]
        
        if len(driver_data) > 5:
            analysis[driver] = {
                'team': team,
                'total_races': len(driver_data['GP'].unique()),
                'avg_stint_length': round(driver_data['StintLength'].mean(), 2),
                'avg_pitstops': round(driver_data['PitStops'].mean(), 2),
                'preferred_compound': driver_data['Compound'].mode().iloc[0] if len(driver_data) > 0 else 'N/A',
                'best_circuits': driver_data.groupby('GP')['StintLength'].mean().nlargest(3).index.tolist(),
                'strategy_patterns': driver_data['Strategy'].value_counts().head(3).to_dict()
            }
    
    return analysis

# Run analysis
driver_analysis = analyze_driver_team_strategies()

print("\n=== TOP DRIVER ANALYSIS ===")
for driver, data in list(driver_analysis.items())[:5]:
    print(f"\n{driver} ({data['team']}):")
    print(f"  Races: {data['total_races']}")
    print(f"  Avg stint: {data['avg_stint_length']} laps")
    print(f"  Avg pitstops: {data['avg_pitstops']}")
    print(f"  Preferred compound: {data['preferred_compound']}")
    print(f"  Best circuits: {data['best_circuits']}")

# Circuit recommendations
def get_circuit_recommendations(circuit_name):
    if circuit_name not in unique_gps:
        return None
        
    circuit_data = merged_df[merged_df['GP'] == circuit_name]
    circuit_info = circuit_df[circuit_df['GP'] == circuit_name].iloc[0]
    
    return {
        'circuit_characteristics': {
            'length': circuit_info['Length'],
            'abrasion': circuit_info['Abrasion'],
            'traction': circuit_info['Traction'],
            'braking': circuit_info['Braking'],
            'grip': circuit_info['Grip'],
            'tyre_stress': circuit_info['TyreStress']
        },
        'historical_avg_stint': round(circuit_data['StintLength'].mean(), 2),
        'historical_avg_pitstops': round(circuit_data['PitStops'].mean(), 2),
        'most_successful_strategy': circuit_data['Strategy'].mode().iloc[0] if len(circuit_data) > 0 else 'N/A',
        'best_compounds': circuit_data['Compound'].value_counts().head(3).index.tolist()
    }

# Get recommendations for key circuits
key_circuits = ['Monaco', 'Bahrain', 'Spain', 'Monza'] if all(c in unique_gps for c in ['Monaco', 'Bahrain', 'Spain']) else unique_gps[:4]

print("\n=== CIRCUIT RECOMMENDATIONS ===")
circuit_recommendations = {}
for circuit in key_circuits:
    rec = get_circuit_recommendations(circuit)
    if rec:
        circuit_recommendations[circuit] = rec
        print(f"\n{circuit}:")
        print(f"  Historical avg stint: {rec['historical_avg_stint']} laps")
        print(f"  Historical avg pitstops: {rec['historical_avg_pitstops']}")
        print(f"  Most successful strategy: {rec['most_successful_strategy']}")
        print(f"  Best compounds: {rec['best_compounds']}")

print("\n✅ ML models trained and saved successfully!")
print("✅ Analysis complete - ready for frontend integration!")