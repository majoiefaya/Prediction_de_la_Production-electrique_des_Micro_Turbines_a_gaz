import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import os

# Set page configuration for a wide layout and descriptive title
st.set_page_config(page_title="Micro Gas Turbine Energy Prediction", layout="wide")

# Function to load and combine static datasets
@st.cache_data
def load_and_combine_datasets():
    """
    Load training and test datasets from static paths and combine them.
    Returns two DataFrames: combined training dataset and combined test dataset.
    """
    try:
        # Define paths to training datasets using os.path.join for cross-platform compatibility
        dataset_ex_1 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../datasets/train/ex_1.csv'))
        dataset_ex_2 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../datasets/train/ex_20.csv'))
        dataset_ex_3 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../datasets/train/ex_21.csv'))
        dataset_ex_4 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../datasets/train/ex_23.csv'))
        dataset_ex_5 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../datasets/train/ex_24.csv'))
        dataset_ex_6 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../datasets/train/ex_9.csv'))
        
        # Combine training datasets into a list
        dataset_list = [dataset_ex_1, dataset_ex_2, dataset_ex_3, dataset_ex_4, dataset_ex_5, dataset_ex_6]
        
        # Concatenate training datasets vertically
        combined_dataset = pd.concat(dataset_list, axis=0)
        combined_dataset.reset_index(drop=True, inplace=True)
        
        # Define paths to test datasets
        dataset_test_1 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../datasets/test/test_1.csv'))
        dataset_test_2 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../datasets/test/test_2.csv'))
        
        # Combine test datasets into a list
        dataset_test_list = [dataset_test_1, dataset_test_2]
        
        # Concatenate test datasets vertically
        combined_dataset_test = pd.concat(dataset_test_list, axis=0)
        combined_dataset_test.reset_index(drop=True, inplace=True)
        
        return combined_dataset, combined_dataset_test
    
    except FileNotFoundError as e:
        st.error(f"Error: One or more dataset files not found. Please ensure all files are in '../datasets/train/' and '../datasets/test/'. Details: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return None, None

# Function to preprocess data
def preprocess_data(df, dataset_type="Training"):
    """
    Preprocess the dataset by converting data types, handling outliers, and adding features.
    Args:
        df (pd.DataFrame): Input DataFrame to preprocess.
        dataset_type (str): Indicates whether it's 'Training' or 'Test' for user feedback.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert 'time' to datetime
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Convert numeric columns to float, handling invalid values
    df['input_voltage'] = pd.to_numeric(df['input_voltage'], errors='coerce')
    df['el_power'] = pd.to_numeric(df['el_power'], errors='coerce')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle outliers in 'el_power' using 3-sigma rule
    mean_el_power = df['el_power'].mean()
    std_el_power = df['el_power'].std()
    z_scores = np.abs((df['el_power'] - mean_el_power) / std_el_power)
    df = df[z_scores < 3]
    
    # Clip 'input_voltage' to a minimum of 3
    df['input_voltage'] = df['input_voltage'].clip(lower=3)
    
    # Add lagged features for time-series prediction
    df['el_power_t-1'] = df['el_power'].shift(1)
    df['el_power_t-2'] = df['el_power'].shift(2)
    df['el_power_t+1'] = df['el_power'].shift(-1)
    
    # Add temporal features from 'time'
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['second'] = df['time'].dt.second
    
    # Drop rows with NaN values resulting from shifts or conversions
    df = df.dropna()
    
    st.info(f"{dataset_type} dataset preprocessed successfully. Rows after preprocessing: {len(df)}")
    return df

# Function to train and evaluate model
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Train a regression model and evaluate its performance.
    Args:
        model: Scikit-learn compatible regression model.
        X_train, X_test: Feature matrices.
        y_train, y_test: Target vectors.
    Returns:
        tuple: Predictions, MSE, RMSE, R² score.
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return y_pred, mse, rmse, r2

# Streamlit app
st.title("Micro Gas Turbine Electrical Energy Prediction")

# Introduction for users
st.markdown("""
### Welcome to the Micro Gas Turbine Energy Prediction App
This application allows you to explore and predict electrical energy output (`el_power`) from micro gas turbine data.
- **Data EDA**: View dataset summaries, visualizations, and correlations.
- **Model Training & Prediction**: Train regression models to predict future `el_power` values and compare their performance.
Navigate using the sidebar to switch between sections.
""")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page", ["Data EDA", "Model Training & Prediction"])

if page == "Data EDA":
    st.header("Exploratory Data Analysis")
    
    # Explanatory session for EDA
    st.markdown("""
    #### Data Exploration Overview
    In this section, we load and preprocess the micro gas turbine datasets, then display:
    - **Dataset Previews**: First few rows of the training and test datasets.
    - **Summary Statistics**: Basic statistics (mean, min, max, etc.) for numerical columns.
    - **Missing Values**: Percentage of missing data in each column.
    - **Visualizations**: Plots to understand trends, distributions, and correlations.
    The datasets are loaded from static paths (`../datasets/train/` and `../datasets/test/`).
    """)
    
    # Load static datasets
    combined_dataset, combined_dataset_test = load_and_combine_datasets()
    
    if combined_dataset is None or combined_dataset_test is None:
        st.error("Cannot proceed with EDA due to dataset loading errors. Check the error message above.")
    else:
        # Preprocess training and test data
        st.markdown("### Preprocessing Datasets")
        st.info("""
        Preprocessing involves:
        - Converting 'time' to datetime format.
        - Ensuring 'input_voltage' and 'el_power' are numeric.
        - Removing duplicates and outliers (using 3-sigma rule for 'el_power').
        - Adding lagged features (e.g., `el_power_t-1`, `el_power_t-2`) for time-series analysis.
        - Extracting temporal features (hour, minute, second).
        """)
        
        processed_dataset = preprocess_data(combined_dataset, "Training")
        processed_dataset_test = preprocess_data(combined_dataset_test, "Test")
        
        # Display dataset previews
        st.subheader("Training Dataset Preview")
        st.write("First 5 rows of the preprocessed training dataset:")
        st.dataframe(processed_dataset.head())
        
        st.subheader("Test Dataset Preview")
        st.write("First 5 rows of the preprocessed test dataset:")
        st.dataframe(processed_dataset_test.head())
        
        # Data Summary
        st.subheader("Data Summary (Training)")
        st.write("Summary statistics for numerical columns in the training dataset:")
        st.write(processed_dataset.describe())
        
        # Missing Values
        st.subheader("Missing Values (Training)")
        st.write("Percentage of missing values in each column (after preprocessing):")
        missing_percent = processed_dataset.isna().mean() * 100
        st.write(missing_percent)
        
        # Visualizations
        st.subheader("Visualizations")
        st.info("""
        The following plots help understand the data:
        - **Line Plot**: Shows `el_power` trends over time.
        - **Histogram**: Displays the distribution of `el_power` with a kernel density estimate (KDE).
        - **Correlation Heatmap**: Visualizes correlations between key variables.
        """)
        
        # Line Plot
        st.write("Trend of El Power over Time (Training)")
        fig = px.line(processed_dataset, x='time', y='el_power', title='Tendance de El Power au fil du temps')
        st.plotly_chart(fig)
        
        # Histogram
        st.write("Distribution of El Power (Training)")
        fig, ax = plt.subplots()
        sns.histplot(processed_dataset['el_power'], bins=50, kde=True, ax=ax)
        ax.set_title('Distribution de El Power avec KDE')
        st.pyplot(fig)
        
        # Correlation Heatmap
        st.write("Correlation Matrix (Training)")
        correlation_matrix = processed_dataset[['input_voltage', 'el_power', 'el_power_t-1', 'el_power_t-2']].corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title('Matrice de corrélation des variables')
        st.pyplot(fig)
        
        # Save processed datasets to session state
        st.session_state['processed_dataset'] = processed_dataset
        st.session_state['processed_dataset_test'] = processed_dataset_test

else:
    st.header("Model Training and Prediction")
    
    # Explanatory session for modeling
    st.markdown("""
    #### Model Training Overview
    In this section, you can:
    - Select a regression model to predict `el_power_t+1` (the next time step's electrical power).
    - View performance metrics (MSE, RMSE, R²).
    - Visualize actual vs. predicted values.
    
    **Available Models**:
    - **Linear Regression**: A simple model assuming linear relationships.
    - **Random Forest**: An ensemble method using multiple decision trees.
    - **CatBoost**: A gradient boosting model optimized for performance.
    
    The models use features like `input_voltage`, `hour`, `minute`, `second`, `el_power_t-1`, and `el_power_t-2`.
    """)
    
    if 'processed_dataset' not in st.session_state or 'processed_dataset_test' not in st.session_state:
        st.error("Please process data in the 'Data EDA' page first.")
    else:
        dataset = st.session_state['processed_dataset']
        dataset_test = st.session_state['processed_dataset_test']
        
        # Feature selection
        features = ['input_voltage', 'hour', 'minute', 'second', 'el_power_t-1', 'el_power_t-2']
        target = 'el_power_t+1'
        
        # Prepare training and test data
        st.info("Preparing data for modeling: Using training dataset for model fitting and test dataset for evaluation.")
        X_train = dataset[features]
        y_train = dataset[target]
        X_test = dataset_test[features]
        y_test = dataset_test[target]
        
        # Normalize features
        st.info("Normalizing features using StandardScaler to ensure consistent scale across variables.")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model selection
        st.subheader("Model Selection")
        model_choice = st.selectbox("Choose a model", ["Linear Regression", "Random Forest", "CatBoost"])
        
        # Initialize model
        if model_choice == "Linear Regression":
            model = LinearRegression()
            st.info("Linear Regression: Fits a linear model to predict `el_power_t+1`.")
        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            st.info("Random Forest: Combines multiple decision trees to improve prediction accuracy.")
        else:
            model = CatBoostRegressor(random_state=42, verbose=0)
            st.info("CatBoost: Uses gradient boosting for robust predictions, especially with time-series data.")
        
        # Train and evaluate
        st.info(f"Training {model_choice} on the training dataset and evaluating on the test dataset...")
        y_pred, mse, rmse, r2 = train_and_evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Display results
        st.subheader("Model Performance")
        st.markdown("""
        The following metrics evaluate the model's performance:
        - **Mean Squared Error (MSE)**: Average squared difference between predictions and actual values.
        - **Root Mean Squared Error (RMSE)**: Square root of MSE, in the same units as `el_power`.
        - **R² Score**: Proportion of variance in `el_power_t+1` explained by the model (closer to 1 is better).
        """)
        st.write(f"**Model:** {model_choice}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")
        
        # Prediction vs Actual Plot
        st.subheader("Actual vs Predicted Values")
        st.info("This plot compares the actual `el_power_t+1` values (blue) with the model's predictions (red).")
        fig, ax = plt.subplots()
        ax.scatter(range(len(y_test)), y_test, color='blue', label='Actual', s=50, alpha=1.0)
        ax.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', s=30, alpha=0.7)
        ax.set_title('Actual vs Predicted (el_power t+1)')
        ax.set_xlabel('Index')
        ax.set_ylabel('el_power')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)