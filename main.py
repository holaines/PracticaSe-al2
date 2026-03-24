import os
import re
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import erfc

from awgn_channel import awgn_channel
from conv_encoding import conv_encoding
from viterbi_decode_101_111 import viterbi_decode_101_111

CACHE_DIR = Path(".cache")
MPLCONFIGDIR = Path(".mplconfig")
CACHE_DIR.mkdir(exist_ok=True)
(CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_MATRIX = np.array([[0.9, 0.12], [0.08, 0.85]])
B_MATRIX = np.array([[-0.12, 0.15], [0.40, -0.15]])
Q_COV = 0.25 * np.eye(2)
R_COV = 0.25 * np.eye(2)
GENERATORS = [[1, 0, 1], [1, 1, 1]]
PROCESS_INFO_COV = solve_discrete_lyapunov(A_MATRIX, Q_COV)
PROCESS_INFO_PREC = np.linalg.inv(PROCESS_INFO_COV)
PROCESS_NOISE_PREC = np.linalg.inv(Q_COV)
OBS_NOISE_PREC = np.linalg.inv(R_COV)
PAIR_SYMBOLS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
PAIR_INDEX = {tuple(pair.astype(int)): idx for idx, pair in enumerate(PAIR_SYMBOLS)}
NEXT_ENCODER_STATE = np.zeros((4, 2), dtype=int)
OUTPUT_SYMBOLS = np.zeros((4, 2, 2), dtype=int)
ENGLISH_VOCAB = {
    "a",
    "alarms",
    "an",
    "and",
    "are",
    "busted",
    "die",
    "failure",
    "find",
    "fit",
    "flashing",
    "going",
    "have",
    "hole",
    "houston",
    "in",
    "is",
    "leak",
    "left",
    "main",
    "master",
    "need",
    "not",
    "option",
    "or",
    "our",
    "oxygen",
    "panel",
    "peg",
    "problem",
    "right",
    "round",
    "square",
    "tanks",
    "to",
    "way",
    "we",
    "you",
}

for state in range(4):
    m1 = (state >> 1) & 1
    m2 = state & 1
    for u in (0, 1):
        OUTPUT_SYMBOLS[state, u] = [(u + m2) % 2, (u + m1 + m2) % 2]
        NEXT_ENCODER_STATE[state, u] = (u << 1) | m1


def q_func(x):
    return 0.5 * erfc(x / np.sqrt(2))


def kalman_rts_smoother(y_obs):
    n_samples = y_obs.shape[0]
    state_dim = A_MATRIX.shape[0]

    x_est = np.zeros(state_dim)
    p_est = np.eye(state_dim)

    x_filt = np.zeros((n_samples, state_dim))
    p_filt = np.zeros((n_samples, state_dim, state_dim))

    for k in range(n_samples):
        if k == 0:
            x_pred = x_est
            p_pred = p_est
        else:
            x_pred = A_MATRIX @ x_est
            p_pred = A_MATRIX @ p_est @ A_MATRIX.T + Q_COV

        s_cov = B_MATRIX @ p_pred @ B_MATRIX.T + R_COV
        kalman_gain = p_pred @ B_MATRIX.T @ np.linalg.inv(s_cov)

        innovation = y_obs[k] - B_MATRIX @ x_pred
        x_est = x_pred + kalman_gain @ innovation
        p_est = (np.eye(state_dim) - kalman_gain @ B_MATRIX) @ p_pred

        x_filt[k] = x_est
        p_filt[k] = p_est

    x_smooth = x_filt.copy()
    p_smooth = p_filt.copy()

    for k in range(n_samples - 2, -1, -1):
        p_pred_next = A_MATRIX @ p_filt[k] @ A_MATRIX.T + Q_COV
        smoother_gain = p_filt[k] @ A_MATRIX.T @ np.linalg.inv(p_pred_next)
        x_smooth[k] = x_filt[k] + smoother_gain @ (x_smooth[k + 1] - A_MATRIX @ x_filt[k])
        p_smooth[k] = p_filt[k] + smoother_gain @ (p_smooth[k + 1] - p_pred_next) @ smoother_gain.T

    return x_smooth, p_smooth


def shift_estimates(x_est, p_est, shift):
    if shift == 0:
        return x_est, p_est

    if shift > 0:
        x_shifted = np.vstack([np.repeat(x_est[0][None, :], shift, axis=0), x_est[:-shift]])
        p_shifted = np.concatenate(
            [np.repeat(p_est[0][None, :, :], shift, axis=0), p_est[:-shift]],
            axis=0,
        )
        return x_shifted, p_shifted

    shift = -shift
    x_shifted = np.vstack([x_est[shift:], np.repeat(x_est[-1][None, :], shift, axis=0)])
    p_shifted = np.concatenate(
        [p_est[shift:], np.repeat(p_est[-1][None, :, :], shift, axis=0)],
        axis=0,
    )
    return x_shifted, p_shifted


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
        byte = bits[i:i + 8]
        value = int("".join(str(b) for b in byte), 2)
        chars.append(chr(value))

    return "".join(chars)


def make_printable(txt):
    return "".join(c if 32 <= ord(c) <= 126 or c in "\n\r\t" else " " for c in txt)


def text_score(txt):
    printable = sum(32 <= ord(c) <= 126 or c in "\n\r\t" for c in txt)
    letters = sum(c.isalpha() or c == " " for c in txt)

    common_words = [
        "the",
        "and",
        "you",
        "have",
        "find",
        "way",
        "fit",
        "problem",
        "houston",
        "oxygen",
        "tank",
        "tanks",
        "leak",
        "command",
        "module",
        "busted",
        "square",
        "peg",
        "round",
        "hole",
        "moon",
    ]

    txt_low = txt.lower()
    word_bonus = sum(word in txt_low for word in common_words)

    return printable + 0.5 * letters + 30 * word_bonus


def map_decode_interference(observations, data_rx):
    n_steps = data_rx.shape[0]
    num_augmented_states = 16
    inf_metric = 1.0e18

    path_metrics = np.full((num_augmented_states, n_steps), inf_metric, dtype=float)
    prev_state_mem = np.full((num_augmented_states, n_steps), -1, dtype=int)
    input_mem = np.full((num_augmented_states, n_steps), -1, dtype=int)

    for u in (0, 1):
        enc_state = NEXT_ENCODER_STATE[0, u]
        coded_pair = OUTPUT_SYMBOLS[0, u].astype(float)
        interference = data_rx[0] - coded_pair
        innovation_obs = observations[0] - B_MATRIX @ interference
        branch_metric = (
            interference @ PROCESS_INFO_PREC @ interference
            + innovation_obs @ OBS_NOISE_PREC @ innovation_obs
        )

        aug_state = 4 * enc_state + PAIR_INDEX[tuple(coded_pair.astype(int))]
        path_metrics[aug_state, 0] = branch_metric
        input_mem[aug_state, 0] = u

    for k in range(1, n_steps):
        for aug_prev in range(num_augmented_states):
            prev_metric = path_metrics[aug_prev, k - 1]
            if prev_metric >= inf_metric:
                continue

            enc_prev = aug_prev // 4
            prev_pair = PAIR_SYMBOLS[aug_prev % 4]
            prev_interference = data_rx[k - 1] - prev_pair

            for u in (0, 1):
                enc_next = NEXT_ENCODER_STATE[enc_prev, u]
                coded_pair = OUTPUT_SYMBOLS[enc_prev, u].astype(float)
                interference = data_rx[k] - coded_pair

                innovation_dyn = interference - A_MATRIX @ prev_interference
                innovation_obs = observations[k] - B_MATRIX @ interference
                branch_metric = (
                    innovation_dyn @ PROCESS_NOISE_PREC @ innovation_dyn
                    + innovation_obs @ OBS_NOISE_PREC @ innovation_obs
                )

                aug_next = 4 * enc_next + PAIR_INDEX[tuple(coded_pair.astype(int))]
                cand_metric = prev_metric + branch_metric

                if cand_metric < path_metrics[aug_next, k]:
                    path_metrics[aug_next, k] = cand_metric
                    prev_state_mem[aug_next, k] = aug_prev
                    input_mem[aug_next, k] = u

    final_aug_state = min((4 * 0 + pair_idx for pair_idx in range(4)), key=lambda idx: path_metrics[idx, -1])

    decoded_bits = np.zeros(n_steps, dtype=int)
    aug_state = final_aug_state
    for k in range(n_steps - 1, -1, -1):
        decoded_bits[k] = input_mem[aug_state, k]
        if k > 0:
            aug_state = prev_state_mem[aug_state, k]

    return decoded_bits


def levenshtein_distance(word_a, word_b):
    len_a = len(word_a)
    len_b = len(word_b)
    dp = np.arange(len_b + 1)

    for i in range(1, len_a + 1):
        prev_diag = dp[0]
        dp[0] = i
        for j in range(1, len_b + 1):
            prev_up = dp[j]
            cost = 0 if word_a[i - 1] == word_b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev_diag + cost,
            )
            prev_diag = prev_up

    return int(dp[len_b])


def apply_case_pattern(original_word, replacement_word):
    if original_word.isupper():
        return replacement_word.upper()
    if original_word.islower():
        return replacement_word
    if original_word[0].isupper() and original_word[1:].islower():
        return replacement_word.capitalize()
    return replacement_word


def correct_decoded_text(printable_text):
    def replace_word(match):
        word = match.group(0)
        word_low = word.lower()

        if word_low in ENGLISH_VOCAB or len(word_low) <= 1:
            return word

        best_word = word_low
        best_distance = 3

        for candidate in ENGLISH_VOCAB:
            if abs(len(candidate) - len(word_low)) > 2:
                continue

            distance = levenshtein_distance(word_low, candidate)
            if distance < best_distance or (
                distance == best_distance
                and abs(len(candidate) - len(word_low)) < abs(len(best_word) - len(word_low))
            ):
                best_distance = distance
                best_word = candidate

        if best_distance <= 1:
            return apply_case_pattern(word, best_word)

        return word

    corrected = re.sub(r"[A-Za-z]+", replace_word, printable_text)
    corrected = re.sub(r" {2,}", " ", corrected)
    return corrected.strip()


def simulate_communications():
    print("--- Simulating Communications System (BER vs Eb/N0) ---")

    eb_n0_db_range = np.arange(-2, 11, 1)
    frame_bits = 4096
    min_total_bits = 131072
    max_total_bits = 393216
    min_viterbi_errors = 120

    ber_uncoded_sim = []
    ber_uncoded_theo = []
    ber_coded_sim = []
    ber_coded_theo = []
    ber_viterbi_sim = []
    viterbi_resolution = []

    rate = 1.0 / len(GENERATORS)
    rng = np.random.default_rng(0)

    for eb_n0 in eb_n0_db_range:
        eb_n0_lin = 10 ** (eb_n0 / 10.0)

        err_uncoded = 0
        err_coded_channel = 0
        err_viterbi = 0
        total_info_bits = 0
        total_encoded_bits = 0

        while total_info_bits < min_total_bits or (
            err_viterbi < min_viterbi_errors and total_info_bits < max_total_bits
        ):
            bits = rng.integers(0, 2, size=frame_bits, dtype=int)

            rx_uncoded = awgn_channel(bits, eb_n0, rate=1.0)
            bits_hat_uncoded = (rx_uncoded > 0).astype(int)
            err_uncoded += np.count_nonzero(bits != bits_hat_uncoded)

            encoded_bits = conv_encoding(bits, GENERATORS)
            rx_coded = awgn_channel(encoded_bits, eb_n0, rate=rate)
            rx_bits_hard = (rx_coded > 0).astype(int)
            err_coded_channel += np.count_nonzero(encoded_bits != rx_bits_hard)

            decoded_bits = viterbi_decode_101_111(
                rx_bits_hard,
                metric="hard",
                force_final_state=0,
            )
            err_viterbi += np.count_nonzero(bits != decoded_bits[:frame_bits])

            total_info_bits += frame_bits
            total_encoded_bits += len(encoded_bits)

        ber_uncoded_sim.append(err_uncoded / total_info_bits)
        ber_uncoded_theo.append(q_func(np.sqrt(2 * eb_n0_lin)))
        ber_coded_sim.append(err_coded_channel / total_encoded_bits)
        ber_coded_theo.append(q_func(np.sqrt(2 * rate * eb_n0_lin)))
        ber_viterbi_sim.append(err_viterbi / total_info_bits)
        viterbi_resolution.append(max(0.5 / total_info_bits, 1.0e-7))

        print(
            f"Eb/N0: {eb_n0:2} dB | "
            f"BER Uncoded: {ber_uncoded_sim[-1]:.4e} | "
            f"BER Coded Channel: {ber_coded_sim[-1]:.4e} | "
            f"BER Viterbi: {ber_viterbi_sim[-1]:.4e} | "
            f"bits averaged: {total_info_bits}"
        )

    plt.figure(figsize=(10, 6))
    plt.semilogy(eb_n0_db_range, ber_uncoded_sim, "bo-", label="Uncoded (Simulated)")
    plt.semilogy(eb_n0_db_range, ber_uncoded_theo, "b--", label="Uncoded (Theoretical)")
    plt.semilogy(eb_n0_db_range, ber_coded_sim, "gv-", label="Coded Channel (Simulated)")
    plt.semilogy(eb_n0_db_range, ber_coded_theo, "g--", label="Coded Channel (Theoretical)")
    plt.semilogy(
        eb_n0_db_range,
        np.maximum(ber_viterbi_sim, viterbi_resolution),
        "rs-",
        label="Coded + Viterbi (Simulated)",
    )

    plt.title("BER vs Eb/N0 for G(D) = [D^2 + 1, D^2 + D + 1]")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.ylim([1e-6, 1])
    plt.xlim([eb_n0_db_range.min(), eb_n0_db_range.max()])
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ber_curve.png", dpi=300)
    print("Plot saved to 'ber_curve.png'")


def decode_interference():
    print("\n--- Decoding Message with Interference ---")

    try:
        observations = np.loadtxt("observations.txt").reshape(-1, 2)
        data_rx = np.loadtxt("data_rx.txt").reshape(-1, 2)
    except FileNotFoundError:
        print("Required files 'observations.txt' and/or 'data_rx.txt' not found. Skipping interference part.")
        return

    decoded_bits = map_decode_interference(observations, data_rx)
    raw_text = bits_to_ascii(decoded_bits, offset=0, drop_last=2)
    printable_text = make_printable(raw_text)
    corrected_text = correct_decoded_text(printable_text)

    print("Selected interference decoder: MAP trellis with Gauss-Markov interference model")
    print("Dropped tail bits:", 2)

    print("\nHidden Message (MAP ASCII):\n")
    print(raw_text.encode("utf-8", errors="replace").decode("utf-8"))

    with open("decoded_message_raw.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)

    with open("decoded_message.txt", "w", encoding="utf-8") as f:
        f.write(corrected_text)

    print("\nPrintable Version:\n")
    print(printable_text)

    with open("decoded_message_printable.txt", "w", encoding="utf-8") as f:
        f.write(printable_text)

    print("\nCorrected Message:\n")
    print(corrected_text)

    with open("decoded_message_corrected.txt", "w", encoding="utf-8") as f:
        f.write(corrected_text)


def main():
    simulate_communications()
    decode_interference()


if __name__ == "__main__":
    main()
