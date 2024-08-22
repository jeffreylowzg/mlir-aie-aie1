import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath):
    data = np.loadtxt(filepath)
    if data.ndim > 1:
        data = data.flatten()
    if len(data) % 2 != 0:
        raise ValueError("The data length should be even to form complex numbers")
    return data

def compute_magnitude_squared(data):
    real_parts = data[0::2]
    imag_parts = data[1::2]
    complex_data = np.array([complex(r, i) for r, i in zip(real_parts, imag_parts)])
    magnitude_squared = np.abs(complex_data)**2
    return magnitude_squared

def get_nth_window(data, window_number, time_bins=32, frequency_bins=25):
    # Calculate the total number of windows
    num_windows = data.size // (time_bins * frequency_bins)
    if window_number >= num_windows:
        raise ValueError(f"Window number {window_number} is out of range. Total windows: {num_windows}")
    
    # Extract the nth window
    start_index = window_number * time_bins * frequency_bins
    end_index = start_index + time_bins * frequency_bins
    window_data = data[start_index:end_index]
    
    # Reshape the data
    window_data_reshaped = window_data.reshape((frequency_bins, time_bins))
    return window_data_reshaped

def plot_spectrogram(data_matlab, data2, magnitude_squared, window_number):
    window_data2 = get_nth_window(data2, window_number)
    window_matlab = get_nth_window(data_matlab, window_number)
    window_magnitude_squared = get_nth_window(magnitude_squared, window_number)

    plt.figure(figsize=(15, 6))

    # Plot data2
    plt.subplot(1, 3, 1)
    plt.imshow(10 * np.log10(window_data2.T), aspect='auto', origin='lower', cmap='viridis')
    plt.title('32x25 Spectrogram (full iron)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(label='Intensity [dB]')

    # Plot magnitude_squared
    plt.subplot(1, 3, 2)
    plt.imshow(10 * np.log10(window_magnitude_squared.T), aspect='auto', origin='lower', cmap='viridis')
    plt.title('32x25 Spectrogram (iron + python)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(label='Intensity [dB]')

    # Plot magnitude_squared
    plt.subplot(1, 3, 3)
    plt.imshow(10 * np.log10(window_matlab.T), aspect='auto', origin='lower', cmap='viridis')
    plt.title('32x25 Spectrogram (matlab)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(label='Intensity [dB]')

    plt.savefig(f'spectrogram_comparison_window_{window_number}.png')  # Save the figure to a file


# Filepath to the data
filepath = "sig0_o_fft.txt"
filepath2 = "iron.txt"
filepath3 = "output_fft_gaussian.txt"

# Load the data
data_python = load_data(filepath)
data_iron = load_data(filepath2)
data_matlab = load_data(filepath3)

# Compute the magnitude squared of the complex data
magnitude_squared = compute_magnitude_squared(data_python)

print(magnitude_squared.shape)
# Compare data2 and magnitude_squared and print differences greater than 1e-6
differences = np.abs(data_iron - magnitude_squared)
differences2 = np.abs(data_matlab - data_iron)
indices = np.where(differences > 1 + 1e-6)
indices2 = np.where(differences2 > 1 + 1e-6)

# Print the values and indices where the difference is greater than 1e-6
for index in zip(*indices):
    print(f"Index: {index}, Data2: {data_iron[index]}, Magnitude Squared: {data_python[index]}, Difference: {differences[index]}")

# Get the nth window (e.g., 9th window)
window_number = 9  # Change this to display different windows

# Plot the spectrogram for the nth window for both data2 and magnitude_squared
plot_spectrogram(data_matlab, data_iron, magnitude_squared, window_number)
