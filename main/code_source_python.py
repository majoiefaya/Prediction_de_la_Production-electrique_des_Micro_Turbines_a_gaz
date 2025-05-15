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
import time

# Set page configuration for a wide layout and descriptive title
st.set_page_config(page_title="Micro Gas Turbine Energy Prediction", layout="wide")

# Function to load and combine static datasets with user selection
@st.cache_data
def load_and_combine_datasets(selected_train_files):
    """
    Load training and test datasets based on user selection.
    Args:
        selected_train_files (list): List of filenames to include for training.
    Returns:
        tuple: Combined training dataset and combined test dataset.
    """
    try:
        dataset_list = []
        train_files_dict = {
            'ex_1.csv': os.path.join(os.path.dirname(__file__), '../datasets/train/ex_1.csv'),
            'ex_20.csv': os.path.join(os.path.dirname(__file__), '../datasets/train/ex_20.csv'),
            'ex_21.csv': os.path.join(os.path.dirname(__file__), '../datasets/train/ex_21.csv'),
            'ex_23.csv': os.path.join(os.path.dirname(__file__), '../datasets/train/ex_23.csv'),
            'ex_24.csv': os.path.join(os.path.dirname(__file__), '../datasets/train/ex_24.csv'),
            'ex_9.csv': os.path.join(os.path.dirname(__file__), '../datasets/train/ex_9.csv')
        }
        
        # Load selected training datasets
        for file in selected_train_files:
            st.write(f"Loading {file}...")
            df = pd.read_csv(train_files_dict[file])
            dataset_list.append(df)
        
        # Concatenate training datasets
        combined_dataset = pd.concat(dataset_list, axis=0)
        combined_dataset.reset_index(drop=True, inplace=True)
        
        # Load test datasets (not user-selectable for simplicity)
        dataset_test_1 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../datasets/test/test_1.csv'))
        dataset_test_2 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../datasets/test/test_2.csv'))
        dataset_test_list = [dataset_test_1, dataset_test_2]
        combined_dataset_test = pd.concat(dataset_test_list, axis=0)
        combined_dataset_test.reset_index(drop=True, inplace=True)
        
        return combined_dataset, combined_dataset_test
    
    except FileNotFoundError as e:
        st.error(f"Error: One or more dataset files not found. Please ensure all files are in '../datasets/train/' and '../datasets/test/'. Details: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return None, None

# Function to preprocess data with user-defined options
def preprocess_data(df, dataset_type="Training", remove_outliers=True, clip_voltage=True):
    """
    Preprocess the dataset with user-configurable options.
    Args:
        df (pd.DataFrame): Input DataFrame to preprocess.
        dataset_type (str): Indicates whether it's 'Training' or 'Test'.
        remove_outliers (bool): Whether to remove outliers using 3-sigma rule.
        clip_voltage (bool): Whether to clip input_voltage to a minimum of 3.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = df.copy()
    
    # Convert 'time' to datetime
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Convert numeric columns to float
    df['input_voltage'] = pd.to_numeric(df['input_voltage'], errors='coerce')
    df['el_power'] = pd.to_numeric(df['el_power'], errors='coerce')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle outliers if selected
    if remove_outliers:
        mean_el_power = df['el_power'].mean()
        std_el_power = df['el_power'].std()
        z_scores = np.abs((df['el_power'] - mean_el_power) / std_el_power)
        df = df[z_scores < 3]
    
    # Clip 'input_voltage' if selected
    if clip_voltage:
        df['input_voltage'] = df['input_voltage'].clip(lower=3)
    
    # Add lagged features
    df['el_power_t-1'] = df['el_power'].shift(1)
    df['el_power_t-2'] = df['el_power'].shift(2)
    df['el_power_t+1'] = df['el_power'].shift(-1)
    
    # Add temporal features
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['second'] = df['time'].dt.second
    
    # Drop rows with NaN values
    df = df.dropna()
    
    st.info(f"{dataset_type} dataset preprocessed successfully. Rows after preprocessing: {len(df)}")
    return df

# Function to train and evaluate model with progress feedback
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Train a regression model and evaluate its performance with progress feedback.
    Returns:
        tuple: Predictions, MSE, RMSE, R² score.
    """
    progress = st.progress(0)
    status_text = st.empty()
    
    # Simulate training progress
    status_text.text("Training model... (Step 1/3)")
    progress.progress(33)
    time.sleep(0.5)
    
    model.fit(X_train, y_train)
    
    status_text.text("Making predictions... (Step 2/3)")
    progress.progress(66)
    y_pred = model.predict(X_test)
    
    status_text.text("Calculating metrics... (Step 3/3)")
    progress.progress(100)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    status_text.text("Training complete!")
    return y_pred, mse, rmse, r2

# Streamlit app
st.title("Micro Gas Turbine Electrical Energy Prediction")

st.markdown("""
### Welcome to the Micro Gas Turbine Energy Prediction App
This interactive application lets you explore and predict electrical energy output (`el_power`) from micro gas turbine data.
- **Data EDA**: Select datasets, customize preprocessing, and explore visualizations.
- **Model Training & Prediction**: Choose models, tune hyperparameters, and compare predictions.
Use the sidebar to navigate between sections and adjust settings.
""")

# Sidebar for navigation and settings
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page", ["Data EDA", "Model Training & Prediction"])

if page == "Data EDA":
    st.header("Exploratory Data Analysis")
    
    st.markdown("""
    #### Data Exploration Overview
    In this section, you can:
    - **Select Training Datasets**: Choose which datasets to include in the analysis.
    - **Customize Preprocessing**: Toggle options like outlier removal and voltage clipping.
    - **Explore Visualizations**: Adjust plot settings to focus on specific features or time ranges.
    """)
    
    # Interactive dataset selection
    st.subheader("Select Training Datasets")
    train_files_options = ['ex_1.csv', 'ex_20.csv', 'ex_21.csv', 'ex_23.csv', 'ex_24.csv', 'ex_9.csv']
    selected_train_files = st.multiselect(
        "Choose training datasets to include (at least one required):",
        options=train_files_options,
        default=train_files_options
    )
    
    if not selected_train_files:
        st.warning("Please select at least one training dataset to proceed.")
    else:
        # Load datasets
        combined_dataset, combined_dataset_test = load_and_combine_datasets(selected_train_files)
        
        if combined_dataset is None or combined_dataset_test is None:
            st.error("Cannot proceed with EDA due to dataset loading errors. Check the error message above.")
        else:
            # Preprocessing options
            st.subheader("Preprocessing Options")
            remove_outliers = st.checkbox("Remove outliers (3-sigma rule)", value=True)
            clip_voltage = st.checkbox("Clip input_voltage to minimum 3", value=True)
            
            st.markdown("### Preprocessing Datasets")
            st.info("""
            Preprocessing steps (based on your selections):
            - Convert 'time' to datetime format.
            - Ensure 'input_voltage' and 'el_power' are numeric.
            - Remove duplicates.
            - Optionally remove outliers and clip 'input_voltage'.
            - Add lagged features (e.g., `el_power_t-1`, `el_power_t-2`).
            - Extract temporal features (hour, minute, second).
            """)
            
            with st.spinner("Preprocessing datasets..."):
                processed_dataset = preprocess_data(combined_dataset, "Training", remove_outliers, clip_voltage)
                processed_dataset_test = preprocess_data(combined_dataset_test, "Test", remove_outliers, clip_voltage)
            
            # Dataset previews
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
            
            # Visualizations with interactivity
            st.subheader("Visualizations")
            st.info("""
            Customize the visualizations:
            - **Line Plot**: Filter the time range for `el_power` trends.
            - **Histogram**: Adjust the number of bins for `el_power` distribution.
            - **Correlation Heatmap**: Select features to include.
            """)
            
            # Line Plot with time range filter
            st.write("Trend of El Power over Time (Training)")
            time_range = st.slider(
                "Select time range to display:",
                min_value=processed_dataset['time'].min().to_pydatetime(),
                max_value=processed_dataset['time'].max().to_pydatetime(),
                value=(processed_dataset['time'].min().to_pydatetime(), processed_dataset['time'].max().to_pydatetime())
            )
            filtered_df = processed_dataset[
                (processed_dataset['time'] >= pd.Timestamp(time_range[0])) &
                (processed_dataset['time'] <= pd.Timestamp(time_range[1]))
            ]
            fig = px.line(filtered_df, x='time', y='el_power', title='Tendance de El Power au fil du temps')
            st.plotly_chart(fig)
            
            # Histogram with adjustable bins
            st.write("Distribution of El Power (Training)")
            num_bins = st.slider("Number of bins for histogram:", min_value=10, max_value=100, value=50)
            fig, ax = plt.subplots()
            sns.histplot(processed_dataset['el_power'], bins=num_bins, kde=True, ax=ax)
            ax.set_title('Distribution de El Power avec KDE')
            st.pyplot(fig)
            
            # Correlation Heatmap with feature selection
            st.write("Correlation Matrix (Training)")
            available_features = ['input_voltage', 'el_power', 'el_power_t-1', 'el_power_t-2', 'hour', 'minute', 'second']
            selected_features = st.multiselect(
                "Select features for correlation matrix:",
                options=available_features,
                default=['input_voltage', 'el_power', 'el_power_t-1', 'el_power_t-2']
            )
            if selected_features:
                correlation_matrix = processed_dataset[selected_features].corr()
                fig, ax = plt.subplots()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
                ax.set_title('Matrice de corrélation des variables')
                st.pyplot(fig)
            else:
                st.warning("Please select at least one feature for the correlation matrix.")
            
            # Save processed datasets to session state
            st.session_state['processed_dataset'] = processed_dataset
            st.session_state['processed_dataset_test'] = processed_dataset_test

else:
    st.header("Model Training and Prediction")
    
    st.markdown("""
    #### Model Training Overview
    In this section, you can:
    - Select a regression model to predict `el_power_t+1`.
    - Adjust model hyperparameters using sliders.
    - View performance metrics (MSE, RMSE, R²).
    - Visualize actual vs. predicted values.
    
    **Available Models**:
    - **Linear Regression**: A simple linear model (no hyperparameters to tune).
    - **Random Forest**: Tune the number of trees and maximum depth.
    - **CatBoost**: Adjust iterations and learning rate.
    """)
    
    if 'processed_dataset' not in st.session_state or 'processed_dataset_test' not in st.session_state:
        st.error("Please process data in the 'Data EDA' page first.")
    else:
        dataset = st.session_state['processed_dataset']
        dataset_test = st.session_state['processed_dataset_test']
        
        st.info("Preparing data for modeling: Using training dataset for model fitting and test dataset for evaluation.")
        features = ['input_voltage', 'hour', 'minute', 'second', 'el_power_t-1', 'el_power_t-2']
        target = 'el_power_t+1'
        
        X_train = dataset[features]
        y_train = dataset[target]
        X_test = dataset_test[features]
        y_test = dataset_test[target]
        
        st.info("Normalizing features using StandardScaler to ensure consistent scale across variables.")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model selection and hyperparameter tuning
        st.subheader("Model Selection and Tuning")
        model_choice = st.selectbox("Choose a model", ["Linear Regression", "Random Forest", "CatBoost"])
        
        # Hyperparameter tuning based on model
        if model_choice == "Linear Regression":
            model = LinearRegression()
            st.info("Linear Regression: Fits a linear model to predict `el_power_t+1`. No hyperparameters to tune.")
        elif model_choice == "Random Forest":
            n_estimators = st.slider("Number of trees in Random Forest:", min_value=10, max_value=200, value=100)
            max_depth = st.slider("Maximum depth of trees:", min_value=5, max_value=50, value=10)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            st.info(f"Random Forest: Using {n_estimators} trees with a maximum depth of {max_depth}.")
        else:
            iterations = st.slider("Number of iterations for CatBoost:", min_value=100, max_value=1000, value=500)
            learning_rate = st.slider("Learning rate for CatBoost:", min_value=0.01, max_value=0.3, value=0.03, step=0.01)
            model = CatBoostRegressor(iterations=iterations, learning_rate=learning_rate, random_state=42, verbose=0)
            st.info(f"CatBoost: Using {iterations} iterations with a learning rate of {learning_rate}.")
        
        # Train and evaluate with progress feedback
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