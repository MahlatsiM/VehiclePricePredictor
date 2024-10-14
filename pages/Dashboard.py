import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def main():
    # Load your data
    df = pd.read_csv("cardekho_dataset.csv", index_col='Unnamed: 0')

    # Convert currency and power units
    df['selling_price'] = df['selling_price'].astype(float) * 0.21
    df['max_power'] = df['max_power'].astype(float) / 1.341

    # Drop unnecessary columns and rename for consistency
    df.rename(columns={'mileage': 'fuel_economy'}, inplace=True)
    df['fuel_economy'] = 100 / df['fuel_economy']
    
    
    #Function to plot univariate analysis using Plotly
    def plot_univariate_analysis(df):
        colors = ['#FFD700', '#001F3F', '#333333', '#A9A9A9']
        numeric_features = df.select_dtypes(include=['float', 'int']).columns
        
        # Create a subplot figure for the distribution of each numeric feature
        fig = make_subplots(rows=3, cols=3, subplot_titles=numeric_features.tolist())
        
        for i, feature in enumerate(numeric_features):
            # Create a density plot for each numeric feature
            fig.add_trace(go.Histogram(
                x=df[feature],
                name=feature,
                histnorm='probability density',  # Normalize to get a density plot
                opacity=0.75,
                xbins=dict(start=df[feature].min(), end=df[feature].max(), size=(df[feature].max() - df[feature].min()) / 30),
                marker_color=colors[i % len(colors)]  # Use colors from the defined palette
            ), row=(i // 3) + 1, col=(i % 3) + 1)  # Determine subplot position
     
        # Update layout for the figure
        fig.update_layout(
            height=800,
            showlegend=False
        )
     
        st.plotly_chart(fig) # Use Streamlit to display the Matplotlib figure

    
    # Function to plot average metrics
    def plot_avg_metrics(df):
    # Define the metrics and update the colors to the desired palette
        metrics = [
            ('max_power', 'Average Max Power for Each Brand', 'Max Power', '#36454F'),  # Charcoal
            ('selling_price', 'Average Selling Price for Each Brand', 'Selling Price', '#0047AB'),  # Deep Blue
            ('km_driven', 'Average Kilometers Driven for Each Brand', 'Kilometers Driven', '#808080'),  # Gray
            ('fuel_economy', 'Average Fuel Economy for Each Brand', 'Fuel Economy', '#FFD700'),  # Gold
            ('engine', 'Average Engine Size for Each Brand', 'Engine Size', '#808080')  # Gray (reused color)
        ]
        
        # Create a figure for subplots
        fig = make_subplots(rows=3, cols=2, subplot_titles=[metric[1] for metric in metrics])
        
        # Iterate over metrics and add each one as a bar plot
        for i, (metric, title, ylabel, color) in enumerate(metrics):
            avg_metric = df.groupby('brand')[metric].mean().reset_index().sort_values(by=metric, ascending=False)
            row, col = divmod(i, 2)
            fig.add_trace(go.Bar(x=avg_metric['brand'], y=avg_metric[metric], name=title, marker_color=color), row=row + 1, col=col + 1)
        
        # Hide unused subplot if the number of metrics is odd
        if len(metrics) % 2 != 0:
            fig.update_xaxes(visible=False, row=3, col=2)
            fig.update_yaxes(visible=False, row=3, col=2)
        
        # Update layout
        fig.update_layout(height=900, width=1200, showlegend=False)
        st.plotly_chart(fig)
        
    
    # Function to plot categorical counts
    def plot_categorical_counts(df, n, m):
        categories = [
            ('fuel_type', 'The Number of Cars for Each Fuel Type', 'Fuel Type'),
            ('seller_type', 'The Number of Cars under Each Seller Type', 'Seller Type'),
            ('transmission_type', 'The Number of Cars under Each Transmission Type', 'Transmission Type')
        ]
    
        total_plots = len(categories)
        
        # Create a figure with specified number of rows and columns
        fig = make_subplots(
            rows=n, 
            cols=m, 
            subplot_titles=[title for _, title, _ in categories], 
            vertical_spacing=0.2, 
            horizontal_spacing=0.15
        )
    
        # Define the color palette (Charcoal, Deep Blue, Gray, Gold)
        colors = ['#36454F', '#0047AB', '#808080', '#FFD700']  
    
        for i, (column, title, xlabel) in enumerate(categories):
            cat_df = df[column].value_counts().reset_index()
            cat_df.columns = [column, 'counts']
            row, col = divmod(i, m)
    
            # Add bar chart to the subplot with the updated color palette
            fig.add_trace(
                go.Bar(
                    x=cat_df[column], 
                    y=cat_df['counts'], 
                    name=title, 
                    marker_color=colors[i % len(colors)]  # Rotate through color palette
                ), 
                row=row + 1, 
                col=col + 1
            )
    
            # Update x and y axes titles
            fig.update_xaxes(title_text=xlabel, row=row + 1, col=col + 1)
            fig.update_yaxes(title_text='Count', row=row + 1, col=col + 1)
    
        # Update layout
        fig.update_layout(
            height=300 * n, 
            width=400 * m,  
            showlegend=False
        )
    
        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    
    # Function to plot correlation heatmap
    def plot_correlation_matrix(df):
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=['float64', 'int64'])
        
        # Compute the correlation matrix
        corr_matrix = numerical_df.corr()
        
        # Define a custom color scale for the correlation matrix
        custom_colors = ["#00008B",
                         "#FFD700",
                         "#808080",
                         "#2A2A2A" ]  # Charcoal, Deep Blue, Gray, Gold
        
        # Plot the heatmap using Plotly
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix of Numerical Variables",
            labels=dict(x="Features", y="Features", color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale=custom_colors
        )
        
        # Add the correlation values on top of the heatmap
        fig.update_traces(text=corr_matrix.values.round(2), texttemplate="%{text}")
        
        # Display the plot in Streamlit
        st.plotly_chart(fig)
        
    def plot_expensive_cars(df):
        # Get the top 20 most expensive cars
        top_20_expensive_cars = df.sort_values(by='selling_price', ascending=False).head(20)
        
        # Define custom colors using the specified palette
        custom_colors = ['#FFD700', '#333333', '#A9A9A9', '#001F3F']
        
        # Create a horizontal bar plot using Plotly
        fig = px.bar(top_20_expensive_cars, 
                     x='selling_price', 
                     y='car_name', 
                     title='Top 20 Most Expensive Cars',
                     orientation='h',
                     color='selling_price', 
                     color_continuous_scale=custom_colors)
        
        # Update layout for better appearance
        fig.update_layout(
            height=800,
            margin=dict(l=150, r=50, t=50, b=50),
            xaxis_title='Price',
            yaxis_title='Car Name',
            yaxis=dict(autorange='reversed')  
        )
        
        # Display the plot in Streamlit
        st.plotly_chart(fig)
    
    def plot_common_brands(df):
        # Get the top 10 most common car models
        car_counts = df['car_name'].value_counts().head(10)
        
        # Convert to DataFrame
        car_counts_df = car_counts.reset_index()
        car_counts_df.columns = ['Car Name', 'Frequency']
        
        # Define custom color sequence from lightest to darkest
        color_sequence = ['#FFD700', '#A9A9A9', '#333333', '#001F3F']
        
        # Create bar plot using Plotly
        fig = px.bar(car_counts_df, 
                     x='Frequency', 
                     y='Car Name', 
                     color='Car Name',  
                     orientation='h',
                     title='Top 10 Most Common Car Brands',
                     color_discrete_sequence=color_sequence[:len(car_counts_df)])
        
        # Update layout for better appearance
        fig.update_layout(
            height=600,  
            margin=dict(l=150, r=50, t=50, b=50),  
            xaxis_title='Frequency',
            yaxis_title='Car Name',
            showlegend=False
        )
        
        # Display the plot in Streamlit
        st.plotly_chart(fig)
    
    # Front page introduction
    st.title("Car Price Analysis Dashboard🏎")
    st.subheader("Explore Key Insights from Car Data")
    st.write("""
    Welcome to the Car Price Prediction Dashboard! This platform provides an interactive way to explore key insights 
    into car features and pricing across various brands. Dive into the data to see trends in car performance, fuel economy, 
    and pricing. Let's discover how different factors influence the value of vehicles, helping you make better-informed decisions.
    """)
    
    # Sidebar Navigation
    st.sidebar.title("📊 Dashboard Navigation")
    st.sidebar.markdown("""
    - [Most Expensive Cars](#most-expensive-cars)
    - [Most Common Cars](#most-common-cars)
    - [Categorical Data Insights](#categorical-data-insights)
    - [Univariate Analysis](#univariate-analysis)
    - [Brand Average Metrics](#brand-average-metrics)
    - [Correlation Analysis](#correlation-analysis)
    """)
    
    # Most Expensive Cars Section (NEW)
    st.write("## 💰 Top 20 Most Expensive Cars")
    plot_expensive_cars(df)
    st.write('''
    ### Insights:
    - **Luxury Cars:** The top of the market is dominated by high-end, luxury vehicles i.e. .
    - **Price Range:** Significant price gaps exist between mass-market and luxury vehicles.
    ''')
    
    # Most Common Cars Section (NEW)
    st.write("## 🚗 Most Common Car Brands")
    plot_common_brands(df)
    st.write('''
    ### Key Insights:
    - **Top Brands:** A handful of brands dominate the market, with significant presence Maruti and Hyundai.
    - **Brand Popularity:** Certain brands have higher volumes, which is driven by affordability and wide model range.
    ''')
    
    # Categorical Count Section
    st.write("## 📊 Categorical Data Insights")
    plot_categorical_counts(df, 2, 2)
    st.write('''
    ### Observations:
    - **Fuel Types:** Petrol and diesel dominate.
    - **Seller Type:** Dealer sales are common.
    - **Transmission Preference:** Majority opt for manual transmission.
    ''')
    
    # Univariate Analysis Section
    st.write("## 🔍 Univariate Analysis of Numerical Features")
    plot_univariate_analysis(df)
    st.write('''
    ### Key Insights:
    - **Vehicle Age and Mileage:** Older vehicles often have higher mileage.
    - **Fuel Economy:** Fairly standardized across most cars.
    - **Engine and Max Power:** Wide variations in engine size and power.
    - **Selling Price:** Skewed, with a few high-priced luxury cars.
    ''')
    
    # Brand Average Metrics Section
    st.write("## 📈 Average Metrics for Each Brand")
    plot_avg_metrics(df)
    st.write('''
    ### Insights:
    - **Performance vs Efficiency:** Luxury brands focus on performance, while others like Renault focus on affordability.
    - **Brand Segmentation:** Distinct targeting of luxury vs mass-market consumers.
    ''')
    
    # Correlation Matrix Section
    st.write("## 🔄 Multivariate Analysis of Numerical Features")
    plot_correlation_matrix(df)
    st.write('''
    ### Key Findings:
    - **Max Power & Engine:** Larger engines produce more power.
    - **Fuel Economy & Power:** Advanced engines show improved efficiency.
    - **Vehicle Age & Price:** Older cars tend to have lower prices.
    ''')
    
    
    
if __name__ == "__main__":
    main()
    