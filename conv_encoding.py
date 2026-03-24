import numpy as np

def conv_encoding(bits, G):
    
    """
    Encode a sequence of bits using a convolutional encoder.
    
    :param bits: 1D numpy array of information bits (0 or 1).
    :param G: list of generator polynomials represented as lists/arrays, 
              e.g., [[1, 0, 1], [1, 1, 1]].
    :return: 1D numpy array of encoded bits.
    """
    bits = np.asarray(bits, dtype=int)
    num_generators = len(G)
    
    # Convolve the info bits with each generator polynomial
    outputs = []
    for g in G:
        outputs.append(np.convolve(bits, g) % 2)
        
    # Interleave the output bits
    outputs = np.array(outputs)
    # The length of each convolution is len(bits) + len(g) - 1
    # Since all generators usually have the same length
    seq_len = outputs.shape[1]
    
    encoded = np.zeros(num_generators * seq_len, dtype=int)
    for i in range(num_generators):
        encoded[i::num_generators] = outputs[i, :]
        
    return encoded
