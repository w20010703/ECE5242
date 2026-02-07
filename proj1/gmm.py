import numpy as np
import matplotlib.pyplot as plt

# MyGMM class defined here (or import from another module)
class MyGMM:
    def __init__(self, k, max_iter=100, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.pi = None
        self.mu = None
        self.sigma = None

    def gaussian_pdf(self, x, mu, sigma):
        d = x.shape[1]
        # Add small value to covariance to avoid singular matrix
        sigma = sigma + np.eye(d) * 1e-6
        det = np.linalg.det(sigma)
        inv = np.linalg.inv(sigma)
        
        norm_coeff = 1.0 / (np.power((2 * np.pi), d/2) * np.sqrt(det) + 1e-9)
        diff = x - mu
        # Compute exponent via matrix multiplication
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
        return norm_coeff * np.exp(exponent)

    def fit(self, x):
        n, d = x.shape
        # 1. Initialize parameters (K-Means or random init)
        self.pi = np.ones(self.k) / self.k
        self.mu = x[np.random.choice(n, self.k, replace=False)]
        self.sigma = np.array([np.eye(d) * np.var(x) for _ in range(self.k)])
        
        log_likelihoods = []

        for i in range(self.max_iter):
            # --- E-Step ---
            probs = np.zeros((n, self.k))
            for k_idx in range(self.k):
                probs[:, k_idx] = self.pi[k_idx] * self.gaussian_pdf(x, self.mu[k_idx], self.sigma[k_idx])
            
            total_prob = np.sum(probs, axis=1, keepdims=True)
            gamma = probs / (total_prob + 1e-9)

            # --- M-Step ---
            nk = np.sum(gamma, axis=0)
            self.pi = nk / n
            
            for k_idx in range(self.k):
                self.mu[k_idx] = np.sum(gamma[:, k_idx:k_idx+1] * x, axis=0) / (nk[k_idx] + 1e-9)
                diff = x - self.mu[k_idx]
                self.sigma[k_idx] = (diff.T @ (gamma[:, k_idx:k_idx+1] * diff)) / (nk[k_idx] + 1e-9)
                self.sigma[k_idx] += np.eye(d) * 1e-5

            # --- Log-Likelihood ---
            curr_log_l = np.sum(np.log(np.sum(probs, axis=1) + 1e-9))
            log_likelihoods.append(curr_log_l)
            print(f"Iteration {i}: Log-Likelihood = {curr_log_l}")

            if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                print("Converged!")
                break
        
        return log_likelihoods

# --- Execution flow ---

# 1. Load labeled data (orange_pixels.npy)
data = np.load('orange_pixels.npy')
print(f"Original data shape: {data.shape}")

# 2. Cast to float64 for numerical precision
data = data.astype(np.float64)

# 3. Create and train model
gmm = MyGMM(k=20, max_iter=100000)
history = gmm.fit(data)

# 4. Plot log-likelihood curve
plt.plot(history)
plt.title('GMM Training Progress (Log-Likelihood)')
plt.xlabel('Iterations')
plt.ylabel('Log-Likelihood')
plt.grid(True)
plt.show()

# 5. Save model parameters (used for prediction later)
np.savez('trained_gmm.npz', pi=gmm.pi, mu=gmm.mu, sigma=gmm.sigma)
print("Model saved as trained_gmm.npz")