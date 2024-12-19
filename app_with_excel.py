from flask import Flask
from flask_socketio import SocketIO
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Paths to model and preprocessing parameters
MODEL_PATH = './model.h5'
PREPROCESSING_PARAMS_PATH = './preprocessing_params.pkl'

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the preprocessing parameters
with open(PREPROCESSING_PARAMS_PATH, 'rb') as f:
    preprocessing_params = pickle.load(f)

scaler = preprocessing_params['scaler']
spatial_filters = preprocessing_params['spatial_filters']
window_size = preprocessing_params['window_size']
step_size = preprocessing_params['step_size']
label_mapping = preprocessing_params['label_mapping']  # If needed later

# Function to preprocess input signals
def preprocess_input_signals(signals):
    """
    Preprocess EEG signals using saved CSP spatial filters and scaler.
    - signals: Raw EEG signals (n_samples, n_channels)
    
    Returns:
        features: CSP features normalized for prediction
    """
    # Windowing the data
    windows = []
    for i in range(0, len(signals) - window_size + 1, step_size):
        window = signals[i:i + window_size]
        windows.append(window)
    
    windows = np.array(windows)
    flattened_windows = windows.reshape(windows.shape[0], -1)
    
    # Apply saved CSP spatial filters
    csp_features = np.dot(flattened_windows, spatial_filters)
    
    # Normalize with the saved scaler
    features = scaler.transform(csp_features)
    return features

def label_to_command(label):
    commands = {
        0: "idle",
        1: "raise both hands",
        2: "raise right hand",
        3: "raise left hand",
    }
    return commands.get(label, "idle")

def predict_and_send_commands(signals):
    features_normalized = preprocess_input_signals(signals)
    y_pred_probs = model.predict(features_normalized)
    y_pred = np.argmax(y_pred_probs, axis=1)

    for pred in y_pred:
        command = label_to_command(pred)
        print(f"Sending Command: {command}")
        socketio.emit("controlRobotArm", command)
        time.sleep(0.5)

# Real-time classification simulation
def simulate_real_time_classification(signal_file_path, model):
    """
    Simulate real-time signal classification and send results to socket.
    - signal_file_path: Path to CSV or Excel file containing EEG signals
    """
    if signal_file_path.endswith('.xlsx'):
        df = pd.read_excel(signal_file_path)
    else:
        df = pd.read_csv(signal_file_path)
    
    # Extract signals
    signals = df[['CH1', 'CH2', 'CH3', 'CH4']].values

    # Preprocess and classify signals in real-time
    predict_and_send_commands(signals)

# ---------------- Main Execution ----------------
if __name__ == '__main__':
    # Path to the input EEG signal file
    input_signal_file = './test.xlsx'

    # Run the classification simulation
    simulate_real_time_classification(
        signal_file_path=input_signal_file,
        model=model
    )
