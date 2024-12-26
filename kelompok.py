import sys
import os
import time
import requests
import numpy as np
import wavio
import sounddevice as sd
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from scipy.io import wavfile
import math

# Fungsi untuk menghitung desibel dari tekanan suara
def hitung_desibel(tekanan_suara, tekanan_referensi=20e-6):
    """
    Menghitung nilai desibel dari tekanan suara.

    :param tekanan_suara: Tekanan suara yang diukur (dalam satuan Pascal)
    :param tekanan_referensi: Tekanan suara referensi (default 20e-6 Pa)
    :return: Nilai desibel (dB)
    """
    if tekanan_suara <= 0:
        raise ValueError("Tekanan suara harus lebih besar dari 0.")
    
    # Menghitung nilai desibel menggunakan rumus 20 * log10(p / p0)
    db = 20 * math.log10(tekanan_suara / tekanan_referensi)
    return db


class EdgeImpulseUploader:
    """Class to handle uploads to Edge Impulse."""
    
    def _init_(self, api_key="ei_f4d6af11f0cddf7f479137a0fcb79fd042f7f5da3995dd13", 
                 api_url="https://ingestion.edgeimpulse.com/api/training/files"):
        self.api_key = api_key
        self.api_url = api_url
        self.label = "suara suasana kelas"  # Default label, can be changed dynamically

    def upload_audio_to_edge_impulse(self, audio_filename):
        try:
            with open(audio_filename, "rb") as f:
                response = requests.post(
                    self.api_url,
                    headers={
                        "x-api-key": self.api_key,
                        "x-label": self.label,
                    },
                    files={"data": (os.path.basename(audio_filename), f, "audio/wav")}, 
                    timeout=30
                )
                if response.status_code == 200:
                    return True, "Uploaded successfully!"
                else:
                    return False, f"Failed with status code: {response.status_code}, response: {response.text}"
        except requests.exceptions.RequestException as e:
            return False, f"Request failed: {e}"


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # Layout setup
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)

        # Title Label
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        self.label_title.setText("<h3 align='center'>Audio Recorder with Edge Impulse Integration</h3>")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 1)

        # Group Box for Parameters
        self.groupBox = QtWidgets.QGroupBox("Parameters", self.centralwidget)
        self.gridLayout_parameters = QtWidgets.QGridLayout(self.groupBox)

        # Input Fields
        self.label_sampling_rate = QtWidgets.QLabel("Sampling Rate:")
        self.lineEdit_sampling_rate = QtWidgets.QLineEdit("16000")
        self.label_update_interval = QtWidgets.QLabel("Update Interval (ms):")
        self.lineEdit_update_interval = QtWidgets.QLineEdit("50")
        self.label_label = QtWidgets.QLabel("Label:")
        self.label_db = QtWidgets.QLabel("Nilai Desibel: 0",)
        font = self.label_db.font()
        font.setPointSize(13)
        self.label_db.setFont(font)
        self.label_db.setAlignment(QtCore.Qt.AlignLeft)
        self.lineEdit_label = QtWidgets.QLineEdit("recording")

        # Buttons
        self.pushButton_record = QtWidgets.QPushButton("Start Recording")
        self.pushButton_replay = QtWidgets.QPushButton("Replay Audio")
        self.pushButton_upload = QtWidgets.QPushButton("Upload to Edge Impulse")
        self.pushButton_replay.setEnabled(False)
        self.pushButton_upload.setEnabled(False)

        # Layout for Parameters
        self.gridLayout_parameters.addWidget(self.label_sampling_rate, 0, 0)
        self.gridLayout_parameters.addWidget(self.lineEdit_sampling_rate, 0, 1)
        self.gridLayout_parameters.addWidget(self.label_update_interval, 1, 0)
        self.gridLayout_parameters.addWidget(self.lineEdit_update_interval, 1, 1)
        self.gridLayout_parameters.addWidget(self.label_label, 2, 0)
        self.gridLayout_parameters.addWidget(self.lineEdit_label, 2, 1)
        self.gridLayout_parameters.addWidget(self.pushButton_record, 3, 0, 1, 2)
        self.gridLayout_parameters.addWidget(self.pushButton_replay, 4, 0, 1, 2)
        self.gridLayout_parameters.addWidget(self.pushButton_upload, 5, 0, 1, 2)
        self.gridLayout_parameters.addWidget(self.label_db, 6, 0, 1, 2)
        self.gridLayout.addWidget(self.groupBox, 1, 0, 1, 1)

        # Plot Widgets for Time and Frequency Domain
        self.plot_widget_time = pg.PlotWidget(self.centralwidget)
        self.plot_widget_time.setBackground('k')
        self.plot_widget_time.setTitle("Time Domain Signal")
        self.plot_widget_time.showGrid(x=True, y=True)
        self.gridLayout.addWidget(self.plot_widget_time, 2, 0, 1, 1)

        self.plot_widget_freq = pg.PlotWidget(self.centralwidget)
        self.plot_widget_freq.setBackground('k')
        self.plot_widget_freq.setTitle("Frequency Domain (DFT)")
        self.plot_widget_freq.showGrid(x=True, y=True)
        self.gridLayout.addWidget(self.plot_widget_freq, 3, 0, 1, 1)

        # Initialize plot data
        self.plot_data_time = self.plot_widget_time.plot(pen=pg.mkPen(color='green', width=1))
        self.plot_data_freq = self.plot_widget_freq.plot(pen=pg.mkPen(color='red', width=1))

        MainWindow.setCentralWidget(self.centralwidget)

        # Initialize parameters
        self.is_recording = False
        self.audio_data = []
        self.audio_file_path = "recorded_audio.wav"
        self.uploader = EdgeImpulseUploader(api_key="ei_01774da3bc0b3abcc6b7efcbc46ec3de7ada9f366e6b6875")

        # Timer for real-time updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)

        # Connect buttons to functions
        self.pushButton_record.clicked.connect(self.toggle_recording)
        self.pushButton_replay.clicked.connect(self.replay_audio)
        self.pushButton_upload.clicked.connect(self.upload_to_edge_impulse)

    def validate_inputs(self):
        try:
            sampling_rate = int(self.lineEdit_sampling_rate.text())
            if sampling_rate <= 0:
                raise ValueError("Sampling rate must be a positive integer.")
            update_interval = int(self.lineEdit_update_interval.text())
            if update_interval <= 0:
                raise ValueError("Update interval must be a positive integer.")
            return True
        except ValueError as e:
            QtWidgets.QMessageBox.warning(None, "Input Error", f"Invalid input: {str(e)}")
            return False

    def toggle_recording(self):
        if not self.is_recording:
            if not self.validate_inputs():
                return
            self.is_recording = True
            self.pushButton_record.setText("Stop Recording")
            self.start_recording()
        else:
            self.is_recording = False
            self.pushButton_record.setText("Start Recording")
            self.stop_recording()

    def start_recording(self):
        self.sampling_rate = int(self.lineEdit_sampling_rate.text())
        self.audio_data = []
        try:
            self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=self.sampling_rate)
            self.stream.start()
            self.timer.start(int(self.lineEdit_update_interval.text()))
        except Exception as e:
            QtWidgets.QMessageBox.warning(None, "Error", f"Failed to start recording: {e}")
            self.is_recording = False
            self.pushButton_record.setText("Start Recording")

    def stop_recording(self):
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
        self.timer.stop()
        self.save_audio()
        self.pushButton_replay.setEnabled(True)
        self.pushButton_upload.setEnabled(True)

    def save_audio(self):
        if self.audio_data:
            audio_data_np = np.concatenate(self.audio_data)
            wavio.write(self.audio_file_path, audio_data_np, self.sampling_rate, sampwidth=2)
            self.pushButton_replay.setEnabled(True)
            self.pushButton_upload.setEnabled(True)

    def replay_audio(self):
        if os.path.exists(self.audio_file_path):
            _, data = wavfile.read(self.audio_file_path)
            sd.play(data, self.sampling_rate)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_data.append(indata.copy())
        self.update_db_value(indata)

    def update_db_value(self, indata):
        rms_value = np.sqrt(np.mean(indata**2))
        if rms_value > 0:
            db_value = hitung_desibel(rms_value)
            self.label_db.setText(f"dB: {db_value:.2f}")

    def update_plot(self):
        if self.audio_data:
            # Time domain plot update
            self.plot_data_time.setData(np.concatenate(self.audio_data)[:, 0])

            # Frequency domain (DFT) plot update
            audio_array = np.concatenate(self.audio_data)
            fft_data = np.fft.fft(audio_array[:, 0])
            freqs = np.fft.fftfreq(len(fft_data), 1/self.sampling_rate)
            self.plot_data_freq.setData(freqs[:len(freqs)//2], np.abs(fft_data[:len(fft_data)//2]))

    def upload_to_edge_impulse(self):
        label = self.lineEdit_label.text()
        if not label:
            QtWidgets.QMessageBox.warning(None, "Error", "Label cannot be empty!")
            return
        self.uploader.label = label
        success, message = self.uploader.upload_audio_to_edge_impulse(self.audio_file_path)
        if success:      
            QtWidgets.QMessageBox.information(None, "Success", message)
        else:
            QtWidgets.QMessageBox.warning(None, "Error", message)


# Run the application
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())