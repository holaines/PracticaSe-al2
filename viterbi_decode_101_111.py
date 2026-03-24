import numpy as np


def _prepare_received_pairs(rx_symbols):
    rx_symbols = np.asarray(rx_symbols)

    if rx_symbols.ndim == 1:
        if len(rx_symbols) % 2 != 0:
            raise ValueError("Input length must be even.")
        return rx_symbols.reshape(-1, 2)

    if rx_symbols.ndim == 2 and rx_symbols.shape[1] == 2:
        return rx_symbols

    raise ValueError("Input must be a flat array with even length or an array of shape (N, 2).")


def viterbi_decode_101_111(
    rx_symbols,
    metric="hard",
    force_final_state=None,
    branch_covariances=None,
):
    """
    Decode the rate-1/2 convolutional code with generators [1,0,1] and [1,1,1].

    Parameters
    ----------
    rx_symbols : array-like
        Received coded sequence. It can be passed as a flat array with even length
        or as an array with shape (N, 2).
    metric : {"hard", "soft"}, optional
        Branch metric used by the decoder.
        - "hard": Hamming distance on 0/1 received bits.
        - "soft": Euclidean distance on real-valued received pairs.
    force_final_state : int or None, optional
        If the encoder is terminated, use 0 to force the traceback to end in the
        all-zero state. When None, the decoder chooses the state with minimum metric.
    branch_covariances : array-like or None, optional
        Optional sequence with shape (N, 2, 2). When provided together with
        ``metric="soft"``, the decoder uses a Mahalanobis branch metric.

    Returns
    -------
    np.ndarray
        Decoded information bits as a 1-D array of 0/1.
    """
    rx_pairs = _prepare_received_pairs(rx_symbols)
    n_steps = rx_pairs.shape[0]

    if metric not in {"hard", "soft"}:
        raise ValueError("metric must be 'hard' or 'soft'.")

    inv_branch_covariances = None
    if branch_covariances is not None:
        branch_covariances = np.asarray(branch_covariances, dtype=float)
        if branch_covariances.shape != (n_steps, 2, 2):
            raise ValueError("branch_covariances must have shape (N, 2, 2).")
        inv_branch_covariances = np.linalg.inv(branch_covariances)

    num_states = 4
    inf_metric = 1.0e18

    pm = np.full((num_states, n_steps + 1), inf_metric, dtype=float)
    prev_state_mem = np.zeros((num_states, n_steps), dtype=int)
    input_mem = np.zeros((num_states, n_steps), dtype=int)
    pm[0, 0] = 0.0

    next_state = np.zeros((num_states, 2), dtype=int)
    out_bits = np.zeros((num_states, 2, 2), dtype=float)

    for state in range(num_states):
        m1 = (state >> 1) & 1
        m2 = state & 1

        for u in (0, 1):
            y1 = (u + m2) % 2
            y2 = (u + m1 + m2) % 2

            out_bits[state, u, :] = [y1, y2]
            next_state[state, u] = (u << 1) | m1

    for k in range(n_steps):
        r = rx_pairs[k]
        weight = None if inv_branch_covariances is None else inv_branch_covariances[k]

        for state in range(num_states):
            if pm[state, k] >= inf_metric:
                continue

            for u in (0, 1):
                s_next = next_state[state, u]
                y = out_bits[state, u, :]

                if metric == "hard":
                    branch_metric = np.sum(r.astype(int) != y.astype(int))
                else:
                    err = r - y
                    branch_metric = err @ err if weight is None else err @ weight @ err

                cand_metric = pm[state, k] + branch_metric

                if cand_metric < pm[s_next, k + 1]:
                    pm[s_next, k + 1] = cand_metric
                    prev_state_mem[s_next, k] = state
                    input_mem[s_next, k] = u

    state = np.argmin(pm[:, n_steps]) if force_final_state is None else force_final_state

    u_hat = np.zeros(n_steps, dtype=int)
    for k in range(n_steps - 1, -1, -1):
        u_hat[k] = input_mem[state, k]
        state = prev_state_mem[state, k]

    return u_hat
