import numpy as np


def sample_normal_distribution(b):
    """Sample from a zero-mean normal distribution with std dev proportional to b."""
    return np.random.normal(0, b)

def sample_motion_model_odometry(u, x_prev, alphas):
    """
    Implements the probabilistic motion model for odometry.

    Args:
        u: tuple of odometry readings (x_bar, y_bar, theta_bar), (x_bar', y_bar', theta_transposed) from time t-1 to t
        x_prev: robot pose at time t-1 as (x, y, θ)
        alphas: motion noise parameters (aplha1, aplha2, aplha3, alpha4)

    Returns:
        New pose x_t = (x', y', θ')
    """
    x_bar_prev, y_bar_prev, theta_bar_prev = u[0]
    x_bar, y_bar, theta_bar = u[1]
    x, y, theta = x_prev

    alpha1, alpha2, alpha3, alpha4 = alphas

    # Step 1: Compute the change in motion
    delta_rot1 = np.arctan2(y_bar - y_bar_prev, x_bar - x_bar_prev) - theta_bar_prev
    delta_trans = np.sqrt((x_bar - x_bar_prev)**2 + (y_bar - y_bar_prev)**2)
    delta_rot2 = theta_bar - theta_bar_prev - delta_rot1

    # Step 2: Add noise to motions
    delta_rot1_hat = delta_rot1 - sample_normal_distribution(alpha1 * abs(delta_rot1) + alpha2 * delta_trans)
    delta_trans_hat = delta_trans - sample_normal_distribution(alpha3 * delta_trans + alpha4 * (abs(delta_rot1) + abs(delta_rot2)))
    delta_rot2_hat = delta_rot2 - sample_normal_distribution(alpha1 * abs(delta_rot2) + alpha2 * delta_trans)

    # Step 3: Compute the new pose
    x_prime = x + delta_trans_hat * np.cos(theta + delta_rot1_hat)
    y_prime = y + delta_trans_hat * np.sin(theta + delta_rot1_hat)
    theta_prime = theta + delta_rot1_hat + delta_rot2_hat

    # Normalize angle
    theta_prime = (theta_prime + np.pi) % (2 * np.pi) - np.pi

    return (x_prime, y_prime, theta_prime)
