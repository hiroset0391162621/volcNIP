from scipy.stats import circmean
import numpy as np

def bootstrap_vm_confidence_interval(data, alpha=0.05, reps=1000):
    """
    Calculate the bootstrap confidence interval for the von Mises mean direction and concentration parameter.

    :param data: array-like, circular data (in radians)
    :param alpha: float, confidence level (default 0.05 for 95% confidence interval)
    :param reps: int, number of bootstrap samples (default 1000)
    :return: tuple, (mean_direction_low, mean_direction_high, kappa_low, kappa_high)
    """
    n = len(data)
    bootstrap_means = []
    bootstrap_kappas = []

    def estimate_kappa(R, n):
        """Estimate the concentration parameter kappa using the average resultant length R."""
        if R < 0.53:
            kappa = 2 * R + R**3 + (5 * R**5) / 6
        elif R < 0.85:
            kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
        else:
            kappa = 1 / (2 * (1 - R))
        return kappa

    for _ in range(reps):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        
        # Compute the mean direction (in radians)
        mean_direction = circmean(sample, high=np.pi, low=-np.pi)
        bootstrap_means.append(mean_direction)
        
        # Compute resultant length R
        C = np.sum(np.cos(sample)) / n
        S = np.sum(np.sin(sample)) / n
        R = np.sqrt(C**2 + S**2)
        
        # Estimate kappa (concentration parameter)
        kappa = estimate_kappa(R, n)
        bootstrap_kappas.append(kappa)

    # Convert results to numpy arrays for easier manipulation
    bootstrap_means = np.array(bootstrap_means)
    bootstrap_kappas = np.array(bootstrap_kappas)
    
    # Compute percentiles for confidence intervals
    mean_direction_low = np.percentile(bootstrap_means, (alpha / 2) * 100)
    mean_direction_high = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    
    kappa_low = np.percentile(bootstrap_kappas, (alpha / 2) * 100)
    kappa_high = np.percentile(bootstrap_kappas, (1 - alpha / 2) * 100)

    return mean_direction_low, mean_direction_high, kappa_low, kappa_high
