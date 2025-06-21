import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading your model and scaler
import xgboost as xgb

def main():
    # Load your trained model and scaler
    scaler = joblib.load(r'mount/src/carpricepredictor/scaler.pkl')  # Update with your scaler's path
    
    # Load the model later
    model = xgb.XGBRegressor()
    model.load_model('Best_model.xgb')
    
    # Function to map brands to numbers
    def map_brands_to_numbers(brands):
        brand_mapping = {brand: index for index, brand in enumerate(brands)}
        return brand_mapping
    
    # List of brands
    brands = ['Audi', 'BMW', 'Bentley', 'Datsun', 'Ferrari', 'Force', 'Ford', 'Honda', 
              'Hyundai', 'ISUZU', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land Rover', 
              'Lexus', 'MG', 'Mahindra', 'Maruti', 'Maserati', 'Mercedes-AMG', 
              'Mercedes-Benz', 'Mini', 'Nissan', 'Porsche', 'Renault', 
              'Rolls-Royce', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']  
    brand_mapping = map_brands_to_numbers(brands)
    
    # Load train sets
    X_train_scaled = np.load('X_train_scaled.npy')
    y_train = np.load('y_train.npy')
    
    # Function to predict car price and return the interval (97.5)
    def predict_car_price_with_interval(model, input_data, scaler, y_train, X_train_scaled, sample_size, std_dev, confidence=1.96):
        # Scale input data
        input_data_scaled = scaler.transform(input_data)
    
        # Predict price
        predicted_price = model.predict(input_data_scaled)[0]
    
        # Calculate margin of error using provided standard deviation and sample size
        margin_of_error = confidence * (std_dev / np.sqrt(sample_size))
        lower_bound = predicted_price - margin_of_error
        upper_bound = predicted_price + margin_of_error
    
        return predicted_price, lower_bound, upper_bound
    
    # Streamlit App Layout
    st.title("🔮 Car Price Prediction Tool")
    st.subheader("Estimate a car's resale value based on its key characteristics")
    st.markdown("This tool allows you to input details about a car and predict its market value. "
                "The price prediction is based on an advanced machine learning model trained on a comprehensive car dataset.")

    # Sidebar Navigation
    st.sidebar.title("🔧 Customize Inputs")
    
    # Select brand
    selected_brand = st.sidebar.selectbox("Select brand:", list(brand_mapping.keys()))
    brand_encoded = brand_mapping.get(selected_brand)
    
    # Input features
    vehicle_year = st.sidebar.number_input("Enter vehicle manufacturing year:", min_value=1900, max_value=2024, value=2015)
    km_driven = st.sidebar.number_input("Enter kilometers driven:", min_value=0, value=50000)
    fuel_economy = st.sidebar.number_input("Enter fuel economy (L/100km):", min_value=0.0, value=6.0)
    engine = st.sidebar.number_input("Enter engine capacity (in CC):", min_value=0, value=1500)
    max_power = st.sidebar.number_input("Enter max power (in kW):", min_value=0, value=75)
    
    # Calculate vehicle_age and power_per_engine
    vehicle_age = 2024 - vehicle_year
    power_per_engine = max_power / engine if engine != 0 else 0
    
    # Categorical inputs
    seller_type = st.sidebar.selectbox("Select seller type:", ["Individual", "Dealer"])
    fuel_type = st.sidebar.selectbox("Select fuel type:", ["Petrol", "Diesel"])
    transmission_type = st.sidebar.selectbox("Select transmission type:", ["Manual", "Automatic"])
    
    # One-hot encode categorical variables
    seller_type_encoded = 1 if seller_type == 'Individual' else 0
    fuel_type_encoded = 0 if fuel_type == 'Diesel' else 1
    transmission_type_encoded = 1 if transmission_type == 'Manual' else 0
    
    # Create DataFrame for input
    input_data = pd.DataFrame({
        'vehicle_age': [vehicle_age],
        'kW_per_CC': [power_per_engine],
        'brand': [brand_encoded],
        'seller_type_Individual': [seller_type_encoded],
        'transmission_type_Manual': [transmission_type_encoded],
        'fuel_economy': [fuel_economy],
        'km_driven': [km_driven],
        'fuel_type_Petrol': [fuel_type_encoded]
        })
    
    if st.button("🔍 Predict Price"):
        # Predict car price with the interval using margin of error calculation
        predicted_price, lower_bound, upper_bound = predict_car_price_with_interval(
            model, input_data, scaler, y_train, X_train_scaled, sample_size=10787, std_dev=77289.01)
        
        # Display the prediction results
        st.write(f"### {vehicle_year} {selected_brand}")
        st.write(f"{km_driven:,} km • {transmission_type} • {fuel_type}")
        st.write("#### Predicted Retail Value")
        st.write(f"# **R{predicted_price:,.2f}**")
        st.write("#### Margin of Error")
        st.write(f"Between **R{lower_bound:,.2f}** and **R{upper_bound:,.2f}**")

    # Style for consistency
    st.write("---")
    st.write("Powered by advanced machine learning algorithms using XGBoost and a custom car dataset.")

if __name__ == "__main__":
    main()
