
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

@st.cache(allow_output_mutation=True)
def load_scaler(scaler_path="scaler.pkl"):  
    """Load the saved scaler."""
    scaler = joblib.load(scaler_path)
    return scaler

@st.cache(allow_output_mutation=True)
def load_trained_model(model_path="best_rnn_model.keras"):
    """Load the saved RNN model."""
    model = load_model(model_path)
    return model

def main():
    st.title("RNN Activity Prediction App")
    st.write("""
    **Instructions:**
    1. Prepare a CSV file with the sensor data.
    2. The CSV should include the following columns:  
       `acceleration_x`, `acceleration_y`, `acceleration_z`, `gyro_x`, `gyro_y`, `gyro_z`, `wrist`
    3. Upload your CSV file below.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            data = pd.read_csv(uploaded_file)
            st.write("### Data Preview:")
            st.dataframe(data.head())

            # Ensure that the expected columns exist
            expected_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'wrist']
            if not all(col in data.columns for col in expected_cols):
                st.error(f"The uploaded CSV does not contain the required columns: {expected_cols}")
                return

            
            features = data[expected_cols].values

            # Load the saved scaler and transform the features
            scaler = load_scaler("scaler.pkl")  
            features_scaled = scaler.transform(features)

            # Reshape the data for the RNN.
            
            features_reshaped = features_scaled.reshape(-1, 7, 1)

            # Load the trained model
            model = load_trained_model("best_rnn_model.keras")

            # Make predictions
            predictions = model.predict(features_reshaped)
            predictions_binary = (predictions >= 0.5).astype(int)

            # Display the predictions
            st.write("### Predictions:")
            st.write(predictions_binary)

            
            data['Prediction'] = predictions_binary
            st.write("### Data with Predictions:")
            st.dataframe(data.head())
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
