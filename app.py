from flask import Flask
from flask_socketio import SocketIO
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import time

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Paths to model and preprocessing parameters
MODEL_PATH = './model_all_data.h5'
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

# Map label index to command
def label_to_command(label):
    commands = {
        0: "idle",
        1: "raise both hands",
        2: "raise left hand",
        3: "raise right hand",
    }
    return commands.get(label, "idle")

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = [0, 1, 2, 3]  # Adjust channels as needed
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 500
        self.window_size = 500
        self.step_size = 250
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='Mindrove Plot', size=(800, 600))

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

    def _init_timeseries(self):
        self.plots = []
        self.curves = []
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)

        dataForSocket = {}
        for count, channel in enumerate(self.exg_channels):
            label = 'CH' + str(channel + 1)
            dataForSocket[label] = data[channel][-self.window_size:]
            
            # Plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 8.0, 30.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.curves[count].setData(data[channel].tolist())

        df = pd.DataFrame(dataForSocket)
        signals = df[['CH1', 'CH2', 'CH3', 'CH4']].values

        # Process and classify signals in real-time, by windows
        for start in range(0, len(signals) - self.window_size + 1, self.step_size):
            window = signals[start:start + self.window_size]
            features_normalized = preprocess_input_signals(window)
            if features_normalized.size == 0:
                continue  # Skip if not enough valid features

            # Predict commands based on model output
            y_pred_probs = model.predict(features_normalized)
            y_pred = np.argmax(y_pred_probs, axis=1)
            for pred in y_pred:
                command = label_to_command(pred)
                print(f"Sending Command: {command}")
                socketio.emit("controlRobotArm", command)
                time.sleep(0.5)

        self.app.processEvents()

def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    params = MindRoveInputParams()

    try:
        board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
        board_shim.prepare_session()
        board_shim.start_stream()
        Graph(board_shim)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=3004, allow_unsafe_werkzeug=True)
    main()
