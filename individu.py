import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SensorSignalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualization of Sensor Signals")
        self.root.geometry("1000x600")

        # Default signal parameters
        self.noise_amplitude = 1.0
        self.noise_frequency = 5.0
        self.time = np.linspace(0, 1, 1000)
        self.signal = np.sin(2 * np.pi * 10 * self.time)  # Default signal
        self.noise = self.generate_noise()
        self.result = self.signal  # Initialize result signal
        self.dft_result = None  # Initialize DFT result

        # Sensor selection variable
        self.selected_sensor = tk.StringVar(value="FSR")

        # Left Panel
        left_frame = tk.Frame(root, width=200, bg="White")
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        bg_color = "LightPink"  # Light pink color
        tk.Label(left_frame, text="Select Sensor", bg=bg_color, font=("Poppins", 12)).pack(pady=10)
        sensors = ["FSR", "GSR", "Heart Rate", "Body Temperature", "GPS"]
        for sensor in sensors:
            tk.Radiobutton(
                left_frame,
                text=sensor,
                bg=bg_color,
                variable=self.selected_sensor,
                value=sensor,
                command=self.update_sensor,
            ).pack(anchor="w")

        tk.Label(left_frame, text="Noise Amplitude", bg="LightPink", font=("Poppins", 10)).pack(pady=5)
        self.noise_amp_slider = ttk.Scale(left_frame, from_=0, to=5, orient=tk.HORIZONTAL, command=self.update_noise)
        self.noise_amp_slider.set(self.noise_amplitude)
        self.noise_amp_slider.pack()

        tk.Label(left_frame, text="Noise Frequency", bg="LightPink", font=("Poppins", 10)).pack(pady=5)
        self.noise_freq_slider = ttk.Scale(left_frame, from_=1, to=50, orient=tk.HORIZONTAL, command=self.update_noise)
        self.noise_freq_slider.set(self.noise_frequency)
        self.noise_freq_slider.pack()

        tk.Label(left_frame, text="Signal Operations", bg="LightPink", font=("Poppins", 12)).pack(pady=10)
        tk.Button(left_frame, text="Add", command=self.add_signals).pack(pady=5, fill="x")
        tk.Button(left_frame, text="Multiply", command=self.multiply_signals).pack(pady=5, fill="x")
        tk.Button(left_frame, text="Convolve", command=self.convolve_signals).pack(pady=5, fill="x")
        tk.Button(left_frame, text="Calculate DFT", command=self.calculate_dft).pack(pady=5, fill="x")
        tk.Button(left_frame, text="Reset", command=self.reset_signals).pack(pady=20, fill="x")

        # Right Panel (Graphs)
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 6))

        # Initialize canvas before calling update_plots
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.update_plots()  # Call this after initializing canvas

    def generate_noise(self):
        return self.noise_amplitude * np.sin(2 * np.pi * self.noise_frequency * self.time)

    def update_noise(self, event=None):
        self.noise_amplitude = self.noise_amp_slider.get()
        self.noise_frequency = self.noise_freq_slider.get()
        self.noise = self.generate_noise()
        self.update_plots()

    def update_sensor(self):
        
        # Update signal based on the selected sensor
        sensor = self.selected_sensor.get()
        if sensor == "Camera":
            self.signal = np.sin(2 * np.pi * 10 * self.time)
        elif sensor == "GPS":
            self.signal = np.cos(2 * np.pi * 5 * self.time)
        elif sensor == "PIR":
            self.signal = np.heaviside(self.time - 0.5, 1)
        elif sensor == "Rainfall":
            self.signal = np.random.uniform(-0.5, 0.5, len(self.time))
        elif sensor == "Sesmik":
            self.signal = np.sin(2 * np.pi * 2 * self.time) + np.random.normal(0, 0.1, len(self.time))
        self.result = self.signal  # Reset result to the new signal
        self.update_plots()

    def add_signals(self):
        self.result = self.signal + self.noise
        self.update_plots()

    def multiply_signals(self):
        self.result = self.signal * self.noise
        self.update_plots()

    def convolve_signals(self):
        self.result = np.convolve(self.signal, self.noise, mode='same')
        self.update_plots()

    def calculate_dft(self):
        self.dft_result = np.abs(np.fft.fft(self.result))[:len(self.time) // 2]
        self.freqs = np.fft.fftfreq(len(self.time), d=(self.time[1] - self.time[0]))[:len(self.time) // 2]
        self.update_plots()

    def reset_signals(self):
        self.update_sensor()  # Reset to current sensor's signal
        self.noise = self.generate_noise()
        self.result = self.signal
        self.dft_result = None
        self.update_plots()

    def update_plots(self):
        for ax in self.axes.flatten():
            ax.clear()

        sensor = self.selected_sensor.get()
        self.axes[0, 0].plot(self.time, self.signal, color="red")
        self.axes[0, 0].set_title(f"{sensor} Signal")
        self.axes[0, 1].plot(self.time, self.noise, color="black")
        self.axes[0, 1].set_title("Noise Signal")

        if self.result is not None:
            self.axes[1, 0].plot(self.time, self.result, color="blue")
            self.axes[1, 0].set_title("Result of Operation")

        if self.dft_result is not None:
            self.axes[1, 1].plot(self.freqs, self.dft_result, color="orange")
            self.axes[1, 1].set_title("DFT Result")
        else:
            self.axes[1, 1].set_title("DFT Result")

        for ax in self.axes.flatten():
            ax.set_xlabel("Time [s]" if "Signal" in ax.get_title() else "Frequency [Hz]")
            ax.set_ylabel("Amplitude [V]")

        self.canvas.draw()

# Run the app
root = tk.Tk()
app = SensorSignalApp(root)
root.mainloop()