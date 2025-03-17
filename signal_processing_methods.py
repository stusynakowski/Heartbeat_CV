def count_boxcars(lst):
    count = 0
    in_boxcar = False
    
    for num in lst:
        if num == 1:
            if not in_boxcar:
                count += 1
                in_boxcar = True
        else:
            in_boxcar = False
    
    return count

from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the signal.
    
    Args:
        signal (list or np.array): The raw signal to filter.
        lowcut (float): Lower cutoff frequency (in Hz).
        highcut (float): Upper cutoff frequency (in Hz).
        fs (float): Sampling frequency (in Hz).
        order (int): Order of the filter.
    
    Returns:
        np.array: The filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Example usage
#fs = 30  # Sampling frequency (e.g., 30 frames per second)
#lowcut = 0.8  # Lower bound of heart rate frequency (0.8 Hz = 48 BPM)
#highcut = 3.0  # Upper bound of heart rate frequency (3.0 Hz = 180 BPM)

# Apply the filter to the raw signal
#filtered_signal = bandpass_filter(signal_raw, lowcut, highcut, fs)

