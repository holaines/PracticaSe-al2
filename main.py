import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

from awgn_channel import awgn_channel
from conv_encoding import conv_encoding
from viterbi_decode_101_111 import viterbi_decode_101_111

def q_func(x):
    return 0.5 * erfc(x / np.sqrt(2))

def kalman_filter(y_obs):
    A = np.array([[0.9, 0.12],
                  [0.08, 0.85]])

    B = np.array([[-0.12, 0.15],
                  [0.40, -0.15]])

    Q = 0.25 * np.eye(2)
    R = 0.25 * np.eye(2)

    N = y_obs.shape[0]
    x_hat = np.zeros((N, 2))

    x_est = np.zeros(2)
    P_est = np.zeros((2, 2))

    for k in range(N):
        x_pred = A @ x_est
        P_pred = A @ P_est @ A.T + Q

        S = B @ P_pred @ B.T + R
        K = P_pred @ B.T @ np.linalg.inv(S)

        innovation = y_obs[k] - B @ x_pred
        x_est = x_pred + K @ innovation
        P_est = (np.eye(2) - K @ B) @ P_pred

        x_hat[k] = x_est

    return x_hat

def bits_to_ascii(bits, offset=0, drop_last=0):
    bits = np.asarray(bits, dtype=int).flatten()

    if offset > 0:
        bits = bits[offset:]

    if drop_last > 0:
        bits = bits[:-drop_last]

    n_bits = (len(bits) // 8) * 8
    bits = bits[:n_bits]

    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        value = int("".join(str(b) for b in byte), 2)
        chars.append(chr(value))

    return "".join(chars)

def text_score(txt):
    printable = sum(32 <= ord(c) <= 126 or c in "\n\r\t" for c in txt)
    letters = sum(c.isalpha() or c == " " for c in txt)

    common_words = [
        "the", "and", "you", "that", "with", "have", "this",
        "from", "your", "for", "are", "was", "find", "square",
        "peg", "hole", "problem"
    ]

    txt_low = txt.lower()
    word_bonus = sum(w in txt_low for w in common_words)

    return printable + 0.5 * letters + 20 * word_bonus

def simulate_communications():
    print("--- Simulating Communications System (BER vs Eb/N0) ---")
    
    # Simulation parameters
    n_bits = 80000  # Number of bits to generate
    # Range of Eb/N0 in dB
    eb_n0_db_range = np.arange(-2, 11, 1)
    
    # Arrays to store Bit Error Rates
    ber_uncoded_sim = []
    ber_uncoded_theo = []
    
    ber_coded_sim = []
    ber_coded_theo = []
    
    ber_viterbi_sim = []
    
    # Convolutional code generators
    G = [[1, 0, 1], [1, 1, 1]]
    rate = 1.0 / len(G)
    
    for eb_n0 in eb_n0_db_range:
        # Convert dB to linear for theoretical calculations
        eb_n0_lin = 10 ** (eb_n0 / 10.0)
        
        # 1. Generate a sequence of bits
        bits = np.random.randint(0, 2, n_bits)
        
        # 2 & 3. Simulate and compute average errors for uncoded transmission
        rx_uncoded = awgn_channel(bits, eb_n0, rate=1.0)
        bits_hat_uncoded = (rx_uncoded > 0).astype(int)
        
        err_uncoded = np.sum(bits != bits_hat_uncoded)
        ber_uncoded_sim.append(err_uncoded / n_bits)
        ber_uncoded_theo.append(q_func(np.sqrt(2 * eb_n0_lin)))
        
        # 4. Encode the bits using the convolutional code
        encoded_bits = conv_encoding(bits, G)
        
        # 5. Coded sequence transmission and error computation (Channel level)
        rx_coded = awgn_channel(encoded_bits, eb_n0, rate=rate)
        rx_bits_hard = (rx_coded > 0).astype(int)
        
        err_coded_channel = np.sum(encoded_bits != rx_bits_hard)
        ber_coded_sim.append(err_coded_channel / len(encoded_bits))
        ber_coded_theo.append(q_func(np.sqrt(2 * rate * eb_n0_lin)))
        
        # 6. Decode the received sequence using Viterbi
        decoded_bits = viterbi_decode_101_111(rx_bits_hard)
        
        # 7. Compute errors after Viterbi decoding
        # Viterbi decoder might trim or process up to n_steps. 
        # Check lengths: decoded_bits length will be length of encoded_bits / 2
        decoded_bits = decoded_bits[:n_bits]
        err_viterbi = np.sum(bits[:len(decoded_bits)] != decoded_bits)
        ber_viterbi_sim.append(err_viterbi / len(decoded_bits))
        
        print(f"Eb/N0: {eb_n0:2} dB | BER Uncoded: {ber_uncoded_sim[-1]:.4e} | BER Coded Channel: {ber_coded_sim[-1]:.4e} | BER Viterbi: {ber_viterbi_sim[-1]:.4e}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.semilogy(eb_n0_db_range, ber_uncoded_sim, 'bo-', label='Uncoded (Simulated)')
    plt.semilogy(eb_n0_db_range, ber_uncoded_theo, 'b--', label='Uncoded (Theoretical)')
    plt.semilogy(eb_n0_db_range, ber_coded_sim, 'gv-', label='Coded Channel (Simulated)')
    plt.semilogy(eb_n0_db_range, ber_coded_theo, 'g--', label='Coded Channel (Theoretical)')
    plt.semilogy(eb_n0_db_range, ber_viterbi_sim, 'rs-', label='Coded + Viterbi (Simulated)')
    
    plt.title('BER vs Eb/N0 for Convolutional Code G = [D^2+1, D^2+D+1]')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.ylim([1e-6, 1])
    plt.xlim([min(eb_n0_db_range), max(eb_n0_db_range)])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("ber_curve.png", dpi=300)
    print("Plot saved to 'ber_curve.png'")
    # plt.show() # Uncomment to show plot during interactive execution

def decode_interference():
    print("\n--- Decoding Message with Interference ---")
    # Load data
    try:
        observations = np.loadtxt("observations.txt").reshape(-1, 2)
        data_rx = np.loadtxt("data_rx.txt").reshape(-1, 2)
    except FileNotFoundError:
        print("Required files 'observations.txt' and/or 'data_rx.txt' not found. Skipping interference part.")
        return

    # 1) Estimate interference using Kalman filter
    x_hat = kalman_filter(observations)

    # 2) Cancel interference
    s_hat = data_rx - x_hat

    # 3) Hard decision: round to nearest bit
    rx_bits_hard = np.clip(np.rint(s_hat), 0, 1).astype(int).flatten()

    # 4) Decode with Viterbi
    decoded_bits = viterbi_decode_101_111(rx_bits_hard)

    # 5) Test offsets and drops to find the hidden message
    best_txt = ""
    best_score = -1e9
    best_offset = 0
    best_drop = 0

    for offset in range(8):
        for drop_last in range(3):
            txt = bits_to_ascii(decoded_bits, offset=offset, drop_last=drop_last)
            sc = text_score(txt)

            if sc > best_score:
                best_score = sc
                best_txt = txt
                best_offset = offset
                best_drop = drop_last

    print("Best offset:", best_offset)
    print("Bits dropped at the end:", best_drop)
    print("\nHidden Message (Decoded):\n")
    print(best_txt.encode('utf-8', errors='replace').decode('utf-8'))

    with open("decoded_message.txt", "w", encoding="utf-8") as f:
        f.write(best_txt)

    # Extra: Print only printable characters
    printable_txt = "".join(c for c in best_txt if 32 <= ord(c) <= 126 or c in "\n\r\t")
    print("\nPrintable Version:\n")
    print(printable_txt)

    with open("decoded_message_printable.txt", "w", encoding="utf-8") as f:
        f.write(printable_txt)

def main():
    simulate_communications()
    decode_interference()

if __name__ == "__main__":
    main()
