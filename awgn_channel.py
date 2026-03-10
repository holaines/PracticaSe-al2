import numpy as np

def awgn_channel(bits, eb_n0_db, rate):
    """
    Simulate the transmission of bits over an AWGN channel.
    Maps bits (0/1) to BPSK symbols (-1/+1) to maintain unit power (Es = 1).
    
    :param bits: 1D numpy array of bits (0 or 1)
    :param eb_n0_db: Signal-to-Noise Ratio (Eb/N0) in decibels
    :param rate: Coding rate R (e.g., 0.5 for a rate 1/2 code, 1.0 for uncoded)
    :return: 1D numpy array of received values
    """
    # Convert SNR from dB to linear scale
    eb_n0_lin = 10 ** (eb_n0_db / 10)
    
    # Calculate Es/N0
    # Es = R * Eb
    es_n0_lin = rate * eb_n0_lin
    
    # Map bits 0 -> -1 and 1 -> 1
    # This BPSK mapping provides unit symbol energy (Es = 1)
    symbols = 2 * np.asarray(bits) - 1
    
    # Variance of the noise sigma^2 = N0 / 2
    # Since Es = 1, N0 = 1 / es_n0_lin, so sigma^2 = 1 / (2 * es_n0_lin)
    sigma = np.sqrt(1.0 / (2.0 * es_n0_lin))
    
    # Generate and add AWGN noise
    noise = np.random.normal(0, sigma, len(symbols))
    received = symbols + noise
    
    return received
