# Car Price Predictor 🏎

A machine learning web application built with Streamlit that predicts car resale values based on vehicle characteristics. The model analyzes factors like brand, mileage, engine specifications, and age to estimate market prices for used cars in the South African market.

## Features

- **Interactive Price Prediction**: Input car details and get instant price estimates with confidence intervals
- **Data Visualization Dashboard**: Explore market trends, brand comparisons, and feature correlations
- **Comprehensive Analysis**: View insights on fuel economy, transmission types, and brand performance metrics
- **User-Friendly Interface**: Clean, intuitive design with guided navigation

## How It Works

1. Users input vehicle details (brand, year, mileage, fuel type, etc.)
2. The XGBoost model processes the data through feature engineering and scaling
3. Predicted retail value is returned with a margin of error based on statistical confidence intervals

## Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Machine Learning**: XGBoost, scikit-learn
- **Model Serialization**: Joblib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/car-price-predictor.git
cd car-price-predictor
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the following files in the root directory:
   - `cardekho_dataset.csv` (training data)
   - `scaler.pkl` (fitted StandardScaler)
   - `Best_model.xgb` (trained XGBoost model)
   - `X_train_scaled.npy` (scaled training features)
   - `Y_train.npy` (training labels)

## Usage

Run the Streamlit app:
```bash
streamlit run Home.py
```

Navigate through the pages:
- **Home**: Introduction and FAQs about car valuation
- **Dashboard**: Explore visualizations and market insights
- **Prediction**: Get price estimates for specific vehicles

## Model Details

- **Algorithm**: XGBoost Regressor
- **Features**: Vehicle age, power-to-weight ratio, brand encoding, fuel economy, mileage, transmission type, seller type, fuel type
- **Confidence Interval**: 95% (1.96 standard deviations)
- **Dataset Size**: 10,787 vehicles

## Project Structure

```
car-price-predictor/
├── Home.py                 # Landing page
├── pages/
│   ├── Dashboard.py        # Data visualization page
│   └── Prediction.py       # Price prediction interface
├── cardekho_dataset.csv    # Training dataset
├── scaler.pkl              # Feature scaler
├── Best_model.xgb          # Trained model
├── X_train_scaled.npy      # Scaled training data
├── Y_train.npy             # Training labels
└── requirements.txt        # Python dependencies
```

## Dataset Information

The dataset includes used car listings with features such as:
- Brand and model
- Manufacturing year
- Kilometers driven
- Fuel type (Petrol/Diesel)
- Transmission type (Manual/Automatic)
- Seller type (Individual/Dealer)
- Engine capacity
- Maximum power output
- Fuel economy

Prices have been converted to South African Rand (ZAR) from the original Indian Rupee values.

## Limitations

- Model trained on historical data and may not reflect real-time market fluctuations
- Limited to brands present in the training dataset
- Price predictions assume average vehicle condition
- Margin of error calculated using training set statistics

## Future Improvements

- Add more South African market data
- Implement model retraining pipeline
- Include vehicle condition as a feature
- Add location-based price adjustments
- Support for more granular model specifications

## Author

Created by Mahlatsi Malise Mashilo as part of a Data Science Capstone Project.

## Acknowledgments

- CarDekho for the original dataset
- South African automotive reviewers: MrHowMuch, Muzi Sambo, Reba S. Cars

## License

This project is for educational purposes. Please respect data usage rights and privacy considerations when deploying or modifying this application.
