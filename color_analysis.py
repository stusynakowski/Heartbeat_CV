import cv2
import numpy as np

def plot_signal_on_image(image, signal, color=(0, 255, 0), thickness=2):
    """
    Plots a signal on the given image.

    :param image: The image on which to plot the signal.
    :param signal: The signal data to plot.
    :param color: The color of the signal line.
    :param thickness: The thickness of the signal line.
    """
    height, width, _ = image.shape
    signal_length = len(signal)
    
    # Normalize signal to fit within the image height
    normalized_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    normalized_signal = (1 - normalized_signal) * height  # Invert and scale to image height

    # Draw the signal on the image
    for i in range(1, signal_length):
        start_point = (int((i - 1) * width / signal_length), int(normalized_signal[i - 1]))
        end_point = (int(i * width / signal_length), int(normalized_signal[i]))
        cv2.line(image, start_point, end_point, color, thickness)

    return image