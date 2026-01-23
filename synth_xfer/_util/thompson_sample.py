import numpy as np

class LinearThompsonSampling:
    def __init__(self, n_features: int, lambda_reg: float = 1.0, v: float = 1.0):
        """
        n_features (d): Dimension of the context vector.
        lambda_reg: Regularization parameter for the covariance matrix.
        v: Scaling factor for the variance (controls exploration).
        """
        self.d = n_features
        self.v = v
        
        # B represents the precision matrix (inverse of covariance approximately)
        # Initialize B as identity matrix scaled by lambda
        self.B = lambda_reg * np.identity(self.d)
        
        # f represents the weighted sum of rewards * contexts
        self.f = np.zeros(self.d)
        
        # theta_hat is the mean of the posterior distribution (Least Squares estimate)
        self.theta_hat = np.zeros(self.d)
        
        # We store B_inv to avoid inverting B at every step (optimization)
        self.B_inv = np.linalg.inv(self.B)

    def select_arm(self, contexts) -> int:
        """
        contexts: A matrix of shape (n_arms, n_features) containing context vectors for each arm.
        Returns: Index of the selected arm.
        """
        # 1. Sample theta_tilde from the posterior distribution N(theta_hat, v^2 * B_inv)
        # We use multivariate_normal for clarity, though Cholesky decomp is faster in prod
        theta_tilde = np.random.multivariate_normal(
            self.theta_hat, 
            (self.v**2) * self.B_inv
        )
        
        # 2. Estimate expected reward for each arm using the sampled theta
        # Shape: (n_arms,)
        estimated_rewards = contexts @ theta_tilde
        
        # 3. Choose the arm with the highest estimated reward
        return np.argmax(estimated_rewards)

    def update(self, context, reward):
        """
        context: The feature vector of the chosen arm (shape: (d,)).
        reward: The observed reward (scalar).
        """
        # Reshape context to ensure it's a column vector effectively for outer product
        # However, for numpy 1D array operations, simple outer product works fine.
        
        # Update B: B = B + x * x^T
        self.B += np.outer(context, context)
        
        # Update f: f = f + r * x
        self.f += reward * context
        
        # Update B_inv and theta_hat
        # Note: In high-performance settings, use Sherman-Morrison formula 
        # to update B_inv iteratively instead of re-inverting.
        self.B_inv = np.linalg.inv(self.B)
        self.theta_hat = self.B_inv @ self.f