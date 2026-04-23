"""
eeg_synthesizer.py
Generates synthetic multi-channel EEG signals for different cognitive states.
Useful for mocking the EEG stream in the backend when hardware is not available.

States modeled:
- 'high_stress': High Beta and Gamma activity.
- 'focused': Moderate Alpha, High Beta activity.
- 'low_focus': High Theta and Alpha activity (drowsy/wandering).
- 'baseline': Balanced activity.
"""

import time
import numpy as np

# Standard EEG frequency bands (Hz)
EEG_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}

def generate_eeg_signal(duration: float = 1.0, fs: int = 256, state: str = "baseline", noise_level: float = 0.5) -> np.ndarray:
    """
    Synthesizes a 1D numpy array representing a single channel of EEG data.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Define relative amplitudes for each band based on the cognitive state
    # Format: (delta, theta, alpha, beta, gamma)
    if state == "high_stress":
        amp_multipliers = (0.5, 0.5, 0.5, 2.5, 2.0)
    elif state == "low_focus":
        amp_multipliers = (1.5, 2.5, 2.0, 0.5, 0.2)
    elif state == "focused":
        amp_multipliers = (0.5, 0.5, 1.2, 2.0, 0.5)
    else:
        # baseline
        amp_multipliers = (1.0, 1.0, 1.0, 1.0, 0.2)
        
    signal = np.zeros_like(t)
    
    # Construct the signal by adding random waves within each frequency band
    bands = list(EEG_BANDS.values())
    for i, (low_freq, high_freq) in enumerate(bands):
        # Generate 3 random components per band for a more organic look
        for _ in range(3):
            freq = np.random.uniform(low_freq, high_freq)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = amp_multipliers[i] * np.random.normal(1.0, 0.2) # Base amplitude + variance
            
            signal += amp * np.sin(2 * np.pi * freq * t + phase)
            
    # Add ambient white noise common in EEG recordings
    white_noise = np.random.normal(0, noise_level, size=t.shape)
    signal += white_noise
    
    return signal

def generate_multichannel_eeg(channels: int = 4, duration: float = 1.0, fs: int = 256, state: str = "baseline") -> np.ndarray:
    """
    Synthesizes a 2D numpy array representing multi-channel EEG data.
    Shape: (channels, samples)
    """
    data = []
    for _ in range(channels):
        channel_signal = generate_eeg_signal(duration=duration, fs=fs, state=state)
        data.append(channel_signal)
    return np.array(data)

def simulate_eeg_stream(state: str = "focused", channels: int = 4, fs: int = 256, chunk_duration: float = 1.0):
    """
    A generator that yields chunks of synthetic EEG data continuously.
    Useful for feeding a real-time signal processing or classification loop.
    """
    print(f"📡 Starting synthetic EEG stream in '{state}' state ({channels} channels, {fs}Hz)...")
    try:
        while True:
            # Sleep to simulate real-time data arrival
            time.sleep(chunk_duration)
            data_chunk = generate_multichannel_eeg(channels, chunk_duration, fs, state)
            yield data_chunk
            
    except KeyboardInterrupt:
        print("\n🛑 EEG stream stopped.")

# ─── Example Usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Generate a single 5-second block of stressed EEG data
    print("Generating a 5-second block of 'high_stress' EEG data (4 channels)...")
    fs = 256
    eeg_data = generate_multichannel_eeg(channels=4, duration=5.0, fs=fs, state="high_stress")
    print(f"Generated data shape: {eeg_data.shape} -> (channels, samples)")
    print(f"Sample data from Channel 1 (first 5 points): {eeg_data[0, :5]}\n")
    
    # 2. Simulate a live stream loop
    # We will simulate a stream that yields 1-second chunks every second. (Ctrl+C to stop)
    print("Simulating live stream for 3 seconds...")
    stream = simulate_eeg_stream(state="low_focus", channels=4, fs=fs, chunk_duration=1.0)
    
    count = 0
    for chunk in stream:
        print(f"Received chunk {count + 1} with shape: {chunk.shape}")
        count += 1
        if count >= 3:
            print("Stopping live stream simulation.")
            break
