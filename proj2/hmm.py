import numpy as np
import pickle
import os

class GestureHMM:
    def __init__(self, n_states, n_observations):
        self.N = n_states        # number of hidden states (e.g. 15)
        self.M = n_observations  # number of observation symbols (K-means clusters, e.g. 75)

        # 1. Initial state distribution pi (length N). Left-to-right: start in state 0.
        self.pi = np.zeros(self.N)
        self.pi[0] = 1.0

        # 2. Transition matrix A (NxN). Strict Left-to-Right.
        self.A = np.zeros((self.N, self.N))
        for i in range(self.N):
            if i == self.N - 1:
                self.A[i, i] = 1.0 
            else:
                self.A[i, i] = 0.5
                self.A[i, i + 1] = 0.5

        # 3. Emission matrix B (NxM). Random init + normalize.
        self.B = np.random.rand(self.N, self.M) + 0.1
        self.B = self.B / self.B.sum(axis=1, keepdims=True)

    def _forward_scaled(self, obs):
        """Scaled forward pass; returns (alpha, scale_factors), log P(obs) = -sum(log(scale))."""
        T = len(obs)
        alpha = np.zeros((T, self.N))
        scale = np.zeros(T)
        alpha[0] = self.pi * self.B[:, obs[0]]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t]
        return alpha, scale

    def _backward_scaled(self, obs, alpha):
        """Backward pass scaled so that at each t, sum_i alpha_t(i)*beta_t(i) = 1."""
        T = len(obs)
        beta = np.zeros((T, self.N))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            raw_beta = self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])
            scale = (alpha[t] * raw_beta).sum()
            if scale > 0:
                beta[t] = raw_beta / scale
            else:
                beta[t] = raw_beta
        return beta

    def _e_step_sequence(self, obs):
        """Compute gamma and xi for one sequence. obs: 1D array of observation indices (0..M-1)."""
        T = len(obs)
        alpha, scale = self._forward_scaled(obs)
        beta = self._backward_scaled(obs, alpha)
        log_prob = -np.sum(np.log(scale))

        gamma = alpha * beta  # (T, N), sums to 1 per t

        xi = np.zeros((T - 1, self.N, self.N))
        for t in range(T - 1):
            xi[t] = alpha[t:t + 1].T * self.A * self.B[:, obs[t + 1]] * beta[t + 1]
            s = xi[t].sum()
            if s > 0:
                xi[t] /= s
        return gamma, xi, log_prob

    def baum_welch(self, observations_list, n_iterations=20):
        """
        observations_list: list of 1D numpy arrays (observation sequences for one gesture).
        Each observation is a cluster index in 0..M-1.
        """
        log_likelihoods = []
        eps = 1e-8  # avoid zeros in B

        for epoch in range(n_iterations):
            # E-step: accumulate sufficient statistics over all sequences
            gamma_sum = np.zeros(self.N)
            xi_sum = np.zeros((self.N, self.N))
            B_num = np.zeros((self.N, self.M))
            pi_num = np.zeros(self.N)
            total_log_prob = 0.0

            for obs in observations_list:
                obs = np.asarray(obs, dtype=np.intp)
                if len(obs) == 0:
                    continue
                gamma, xi, log_prob = self._e_step_sequence(obs)
                total_log_prob += log_prob
                gamma_sum += gamma.sum(axis=0)
                xi_sum += xi.sum(axis=0)
                pi_num += gamma[0]
                for t in range(len(obs)):
                    B_num[:, obs[t]] += gamma[t]

            log_likelihoods.append(total_log_prob)

            # M-step: update pi, A, B
            if gamma_sum.sum() > 0:
                # pi from gamma at t=0 (averaged over sequences)
                self.pi = pi_num / (pi_num.sum() + eps)
                self.pi = np.clip(self.pi, 0, 1)
                self.pi /= self.pi.sum()

                # A
                A_den = xi_sum.sum(axis=1, keepdims=True)
                A_den[A_den == 0] = 1
                self.A = xi_sum / A_den
                self.A = np.clip(self.A, 0, 1)
                self.A /= self.A.sum(axis=1, keepdims=True)

                # B (add eps only once to avoid zeros, then renormalize)
                B_den = gamma_sum[:, np.newaxis]
                B_den[B_den == 0] = 1
                self.B = B_num / B_den
                self.B = self.B + eps
                self.B = self.B / self.B.sum(axis=1, keepdims=True)

        return log_likelihoods


if __name__ == '__main__':
    # Load discrete sequences (from train_data.py): gesture_name -> list of 1D arrays
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(script_dir, 'gesture_sequences.pkl')
    if not os.path.exists(pkl_path):
        raise SystemExit(f"Run train_data.py first to create {pkl_path}")

    with open(pkl_path, 'rb') as f:
        gesture_sequences = pickle.load(f)

    n_states = 12       # hidden states per gesture (e.g. 15)
    n_observations = 90 # must match K-means num_clusters in train_data.py

    hmm_models = {}
    for gesture_name, obs_list in gesture_sequences.items():
        print(f"Training HMM for gesture: {gesture_name} ({len(obs_list)} sequences)")
        hmm = GestureHMM(n_states=n_states, n_observations=n_observations)
        log_likelihoods = hmm.baum_welch(obs_list, n_iterations=20)
        hmm_models[gesture_name] = hmm
        print(f"  Final log-likelihood: {log_likelihoods[-1]:.2f}")

    # Save trained HMMs for use at test time
    models_path = os.path.join(script_dir, 'hmm_models.pkl')
    with open(models_path, 'wb') as f:
        pickle.dump(hmm_models, f)
    print(f"Saved {len(hmm_models)} HMMs to {models_path}")