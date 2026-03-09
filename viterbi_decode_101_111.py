import numpy as np


def viterbi_decode_101_111(rx_bits):
    """
    Hard-decision Viterbi decoder for the rate-1/2 convolutional code
    with generators:
        g1 = [1, 0, 1]
        g2 = [1, 1, 1]

    Constraint length K = 3, so there are 2^(K-1) = 4 states.

    Parameters
    ----------
    rx_bits : array-like of 0/1
        Received coded bits. Length must be even.

    Returns
    -------
    u_hat : np.ndarray
        Decoded information bits as a 1-D array of 0/1.
    """
    rx_bits = np.asarray(rx_bits, dtype=int).flatten()

    if len(rx_bits) % 2 != 0:
        raise ValueError("Input length must be even.")

    n_steps = len(rx_bits) // 2
    num_states = 4
    inf_metric = 10**9

    # Path metrics
    pm = np.full((num_states, n_steps + 1), inf_metric, dtype=float)

    # Survivor memory
    prev_state_mem = np.zeros((num_states, n_steps), dtype=int)
    input_mem = np.zeros((num_states, n_steps), dtype=int)

    # Start in all-zero state
    pm[0, 0] = 0

    # Precompute transitions and outputs
    # State encoding:
    #   0 -> [0,0]
    #   1 -> [0,1]
    #   2 -> [1,0]
    #   3 -> [1,1]
    #
    # Here:
    #   m1 = u(k-1), m2 = u(k-2)
    next_state = np.zeros((num_states, 2), dtype=int)
    out_bits = np.zeros((num_states, 2, 2), dtype=int)

    for s in range(num_states):
        m1 = (s >> 1) & 1
        m2 = s & 1

        for u in (0, 1):
            # g1 = [1,0,1] => y1 = u xor m2
            # g2 = [1,1,1] => y2 = u xor m1 xor m2
            y1 = (u + m2) % 2
            y2 = (u + m1 + m2) % 2

            out_bits[s, u, :] = [y1, y2]

            # Next state = [u, m1]
            s_next = (u << 1) | m1
            next_state[s, u] = s_next

    # Forward recursion
    for k in range(n_steps):
        r = rx_bits[2 * k: 2 * k + 2]

        for s in range(num_states):
            if pm[s, k] >= inf_metric:
                continue

            for u in (0, 1):
                s_next = next_state[s, u]
                y = out_bits[s, u, :]

                # Hamming distance branch metric
                branch_metric = np.sum(r != y)
                cand_metric = pm[s, k] + branch_metric

                if cand_metric < pm[s_next, k + 1]:
                    pm[s_next, k + 1] = cand_metric
                    prev_state_mem[s_next, k] = s
                    input_mem[s_next, k] = u

    # Traceback
    # If encoder was terminated, you can force final state = 0.
    state = np.argmin(pm[:, n_steps])

    u_hat = np.zeros(n_steps, dtype=int)
    for k in range(n_steps - 1, -1, -1):
        u_hat[k] = input_mem[state, k]
        state = prev_state_mem[state, k]

    return u_hat