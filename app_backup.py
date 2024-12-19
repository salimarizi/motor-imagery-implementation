from flask import Flask
from flask_socketio import SocketIO
import numpy as np
import time
import threading
import pickle
import tensorflow as tf

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load your model and features
model = tf.keras.models.load_model("model.h5")
with open("preprocessing_params.pkl", "rb") as f:
    preprocessing_params = pickle.load(f)

# Map labels to commands for the frontend
def label_to_command(label):
    commands = {
        0: "idle",
        1: "raise both hands",
        2: "raise right hand",
        3: "raise left hand",
    }
    return commands.get(label, "idle")

# Function to run predictions and send commands
def predict_and_send_commands():
    features_normalized = preprocessing_params.get("normalization_params")
    y_pred_probs = model.predict(features_normalized)
    y_pred = np.argmax(y_pred_probs, axis=1)

    for pred in y_pred:
        command = label_to_command(pred)
        print(f"Sending Command: {command}")
        socketio.emit("controlRobotArm", command)
        time.sleep(0.5)

# Background thread to run the predictions
def background_task():
    while True:
        predict_and_send_commands()
        time.sleep(5)  # Optional: Run predictions every 5 seconds or as needed

# Run Flask with SocketIO
if __name__ == "__main__":
    thread = threading.Thread(target=background_task)
    thread.daemon = True
    thread.start()
    socketio.run(app, host="0.0.0.0", port=3004, allow_unsafe_werkzeug=True)
