import numpy as np
import matplotlib.pyplot as plt

def ewma(data, window):
    """
    Calculate Exponential Weighted Moving Average (EWMA) for the given data with the specified window size.
    
    Parameters:
        data (numpy.ndarray): Input data.
        window (int): Window size for EWMA.
    
    Returns:
        numpy.ndarray: EWMA values.
    """
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    return np.convolve(data, weights, mode='valid')

# Example usage:
data = np.random.rand(100)  # Generate random data
window = 2  # Window size for EWMA
ewma_values = ewma(data, window)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data, label='Original Data', color='blue', alpha=0.5)
plt.plot(np.arange(window - 1, len(data)), ewma_values, label=f'EWMA ({window}-period)', color='red')
plt.title('Exponential Weighted Moving Average (EWMA)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
