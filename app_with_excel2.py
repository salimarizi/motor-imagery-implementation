from flask import Flask
from flask_socketio import SocketIO
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import threading
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
label_mapping = preprocessing_params['label_mapping']

# Function to preprocess input signals
def preprocess_input_signals(signals):
    if len(signals) < window_size:
        print(f"Not enough data to form even one window. Signal length: {len(signals)}")
        return np.array([])

    flattened_window = signals.reshape(1, -1)
    csp_features = np.dot(flattened_window, spatial_filters)

    # Apply saved CSP spatial filters
    features = scaler.transform(csp_features)
    return features

def label_to_command(label):
    commands = {
        0: "idle",
        1: "raise both hands",
        2: "raise left hand",
        3: "raise right hand",
    }
    return commands.get(label, "idle")

def predict_and_send_commands_window(signals):
    features_normalized = preprocess_input_signals(signals)
    if features_normalized.size == 0:
        return

    y_pred_probs = model.predict(features_normalized)
    y_pred = np.argmax(y_pred_probs, axis=1)

    for pred in y_pred:
        command = label_to_command(pred)
        print(f"Sending Command: {command}")
        socketio.emit("controlRobotArm", command)
        time.sleep(0.5)

# Real-time classification simulation
def simulate_real_time_classification(signal_file_path):
    if signal_file_path.endswith('.xlsx'):
        df = pd.read_excel(signal_file_path)
    else:
        df = pd.read_csv(signal_file_path)

    # Extract signals (for example, CH1, CH2, CH3, CH4 columns)
    signals = df[['CH1', 'CH2', 'CH3', 'CH4']].values

    # Process and classify signals in real-time, by windows
    for start in range(0, len(signals) - window_size + 1, step_size):
        window = signals[start:start + window_size]
        predict_and_send_commands_window(window)

# Background thread to run classification
def background_task():
    input_signal_file = './test.xlsx'  # Path to your signal file
    while True:
        simulate_real_time_classification(signal_file_path=input_signal_file)
        time.sleep(5)  # Optional: Run predictions every 5 seconds

# Main execution
if __name__ == '__main__':
    thread = threading.Thread(target=background_task)
    thread.daemon = True
    thread.start()
    socketio.run(app, host="0.0.0.0", port=3004)
