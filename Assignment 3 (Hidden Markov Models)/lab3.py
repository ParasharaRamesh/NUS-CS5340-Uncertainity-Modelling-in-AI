""" CS5340 Lab 3: Hidden Markov Models
See accompanying PDF for instructions.

Name: Parashara Ramesh
Email: e1216292@u.nus.edu
Student ID: A0285647M
"""
import numpy as np
import scipy.stats
from scipy.special import softmax
from sklearn.cluster import KMeans


def initialize(n_states, x):
    """Initializes starting value for initial state distribution pi
    and state transition matrix A.

    A and pi are initialized with random starting values which satisfies the
    summation and non-negativity constraints.
    """
    seed = 5340
    np.random.seed(seed)

    pi = np.random.random(n_states)
    A = np.random.random([n_states, n_states])

    # We use softmax to satisify the summation constraints. Since the random
    # values are small and similar in magnitude, the resulting values are close
    # to a uniform distribution with small jitter
    pi = softmax(pi)
    A = softmax(A, axis=-1)

    # Gaussian Observation model parameters
    # We use k-means clustering to initalize the parameters.
    x_cat = np.concatenate(x, axis=0)
    kmeans = KMeans(n_clusters=n_states, random_state=seed).fit(x_cat[:, None])
    mu = kmeans.cluster_centers_[:, 0]
    std = np.array([np.std(x_cat[kmeans.labels_ == l]) for l in range(n_states)])
    phi = {'mu': mu, 'sigma': std}

    return pi, A, phi


"""E-step"""


def e_step(x_list, pi, A, phi):
    """ E-step: Compute posterior distribution of the latent variables,
    p(Z|X, theta_old). Specifically, we compute
      1) gamma(z_n): Marginal posterior distribution, and
      2) xi(z_n-1, z_n): Joint posterior distribution of two successive
         latent states

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        pi (np.ndarray): Current estimated Initial state distribution (K,)
        A (np.ndarray): Current estimated Transition matrix (K, K)
        phi (Dict[np.ndarray]): Current estimated gaussian parameters

    Returns:
        gamma_list (List[np.ndarray]), xi_list (List[np.ndarray])
    """
    n_states = pi.shape[0]
    gamma_list = [np.zeros([len(x), n_states]) for x in x_list]
    xi_list = [np.zeros([len(x) - 1, n_states, n_states]) for x in x_list]

    """ YOUR CODE HERE
    Use the forward-backward procedure on each input sequence to populate 
    "gamma_list" and "xi_list" with the correct values.
    Be sure to use the scaling factor for numerical stability.
    """
    # 1. calculate scaled alpha
    alphas_hat, c_n = Estep.calculate_scaled_alphas_and_c(x_list, pi, phi, A)

    # 2.  calculate scaled beta
    # betas_hat = Estep.calculate_scaled_betas(x_list, pi, phi, A, c_n)

    # 3. calculate gamma
    # gamma_list = Estep.calculate_gamma(alphas_hat, betas_hat)

    # 4 calculate xi_list
    # xi_list = Estep.calculate_spring(alphas_hat, betas_hat, c_n, pi, phi, A)

    return gamma_list, xi_list


class Estep:
    @staticmethod
    def multivariate_normal_pdf(x, mu, cov):
        k = mu.shape[0]
        det_cov = np.linalg.det(cov)
        inv_cov = np.linalg.inv(cov)
        diff = x - mu

        exponent = -0.5 * np.dot(np.dot(diff, inv_cov), diff)
        coefficient = 1.0 / (np.sqrt((2 * np.pi) ** k * det_cov))

        return coefficient * np.exp(exponent)

    @staticmethod
    def calculate_scaled_alphas_and_c(x_list, pi, phi, A):
        xs = np.array(x_list) #Shape (O, N)
        mu = phi["mu"] #Shape (K)
        sigma = phi["sigma"] #Shape (K)

        O = xs.shape[0]
        N = xs.shape[1]
        K = mu.shape[0]

        alphas_hat = [] #should eventually be a shape of (O,N,K)
        c = [] #should eventually be a shape of (O, N)

        for o in range(O): # for each observation
            # base case for this observation

            # using this base case perform tabulation to find alpha tilde

            # from alpha tilde find out the cn and alpha_hat(zn)

            # store in the arrays


            pass

        #return the final scaled alphas and the constants
        return np.array(alphas_hat), np.array(c)


"""M-step"""


def m_step(x_list, gamma_list, xi_list):
    """M-step of Baum-Welch: Maximises the log complete-data likelihood for
    Gaussian HMMs.

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        gamma_list (List[np.ndarray]): Marginal posterior distribution
        xi_list (List[np.ndarray]): Joint posterior distribution of two
          successive latent states

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.
    """

    n_states = gamma_list[0].shape[1]
    pi = np.zeros([n_states])
    A = np.zeros([n_states, n_states])
    phi = {'mu': np.zeros(n_states),
           'sigma': np.zeros(n_states)}

    """ YOUR CODE HERE
    Compute the complete-data maximum likelihood estimates for pi, A, phi.
    """

    # 1. Finding new pi
    pi = Mstep.find_new_pi(gamma_list, pi)

    # 2. Finding new mu
    phi["mu"] = Mstep.find_new_mu(gamma_list, x_list)

    # 3. Finding new A
    A = Mstep.find_new_A(xi_list, A)

    # 4. Finding new sigma
    phi["sigma"] = Mstep.find_new_sigma(gamma_list, x_list, phi["mu"])

    return pi, A, phi


""" Functions for calculating each new parameter"""


class Mstep:
    @staticmethod
    def find_new_pi(gamma_list, pi):
        # Getting (Obs, N, k) -> (Obs, k) ( picking it for the N = 1)
        gamma_1s = np.vstack([gamma_obs[0] for gamma_obs in gamma_list])
        # Numerator: Adding across all observations to get from (Obs, k) -> (k)
        pi_num = np.sum(gamma_1s, axis=0)
        # Denominator: Summing across the k to get the normalizer
        pi_denom = np.sum(pi_num, axis=0)
        # Computing pi
        pi = pi_num / pi_denom
        return pi

    @staticmethod
    def find_new_mu(gamma_list, x_list):
        gammas = np.array(gamma_list)
        # a. Finding Denominator
        # Find sum across all time steps (Obs, N, k) -> (Obs, k)
        mu_denom = np.sum(gammas, axis=1)
        # Do this across all observations (Obs, k) -> (k)
        mu_denom = np.sum(mu_denom, axis=0)

        # b. Finding Numerator
        xs = np.array(x_list)
        # find the product with each x i.e. (Obs, N, k) x (Obs,N,1) -> (Obs, N, K) {the 1 is added using np.newaxis}
        mu_num = gammas * xs[:, :, np.newaxis]
        # Do summation over all timesteps to get from (Obs, N, K) -> (Obs, K)
        mu_num = np.sum(mu_num, axis=1)
        # Do summation over all observations to get from (Obs, K) -> (K)
        mu_num = np.sum(mu_num, axis=0)

        # c. finding mu
        return mu_num / mu_denom

    @staticmethod
    def find_new_A(xi_list, A):
        '''
        Fill the Ajk values one by one

        @param xi_list:
        @param A:
        @return:
        '''

        xis = np.array(xi_list)
        K = A.shape[0]
        O = xis.shape[0]
        N = xis.shape[1] + 1

        xi_n_sums_per_obs = []
        for xi_obs in xis:  # Shape (Obs, N-1 , K, K)
            # xi_obs is Shape (N-1 , K, K)
            xi_n_sum_per_obs = np.zeros((K, K))
            for xi_n in xi_obs:
                # xi_n is Shape (K, K)
                xi_n_sum_per_obs += xi_n
            xi_n_sums_per_obs.append(xi_n_sum_per_obs)
        xi_n_sums_per_obs = np.array(xi_n_sums_per_obs)

        # Numerator
        A_num = np.sum(xi_n_sums_per_obs, axis=0)

        # Denominator
        A_denom = np.sum(A_num, axis=1)
        A_denom = A_denom[:, np.newaxis]

        # Compute A
        A = A_num / A_denom

        return A

    @staticmethod
    def find_new_sigma(gamma_list, x_list, mu):
        gammas = np.array(gamma_list)
        xs = np.array(x_list)

        O = xs.shape[0]
        N = xs.shape[1]
        K = mu.shape[0]

        covariance = np.zeros((K))

        # finding numerator
        for k in range(K):
            num_across_obs = 0
            denom_across_obs = 0
            for o in range(O):
                num_across_N = 0
                denom_across_N = 0
                for n in range(N):
                    granular_gamma = gammas[o, n, k]
                    x_n = xs[o, n]
                    mu_k = mu[k]
                    diff = x_n - mu_k
                    diff_transpose = diff.T
                    granular_product = granular_gamma * diff * diff_transpose
                    num_across_N += granular_product
                    denom_across_N += granular_gamma
                num_across_obs += num_across_N
                denom_across_obs += denom_across_N
            covariance[k] = num_across_obs / denom_across_obs

        # Take sqrt from covariance to get the sigma (i.e. std deviation)
        sigma = np.sqrt(covariance)

        return sigma


"""Putting them together"""


def fit_hmm(x_list, n_states):
    """Fit HMM parameters to observed data using Baum-Welch algorithm

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        n_states (int): Number of latent states to use for the estimation.

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Time-independent stochastic transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.

    """

    # We randomly initialize pi and A, and use k-means to initialize phi
    # Please do NOT change the initialization function since that will affect
    # grading
    pi, A, phi = initialize(n_states, x_list)

    """ YOUR CODE HERE
     Populate the values of pi, A, phi with the correct values. 
    """
    threshold = 1e-4

    while True:
        gamma_list, xi_list = e_step(x_list, pi, A, phi)
        pi_new, A_new, phi_new = m_step(x_list, gamma_list, xi_list)

        has_pi_converged = np.all(np.isclose(pi, pi_new, threshold))
        has_A_converged = np.all(np.isclose(A, A_new, threshold))
        has_phi_mu_converged = np.all(np.isclose(phi["mu"], phi_new["mu"], threshold))
        has_phi_sigma_converged = np.all(np.isclose(phi["sigma"], phi_new["sigma"], threshold))

        pi, A, phi = pi_new, A_new, phi_new

        if has_pi_converged and has_A_converged and has_phi_mu_converged and has_phi_sigma_converged:
            break

    return pi, A, phi


if __name__ == '__main__':
    # Create sample arrays G and d
    G = np.random.rand(200, 8, 3)  # G shape: (200, 8, 3)
    d = np.random.rand(200, 8, 3)  # d shape: (200, 8, 3)

    # Step 1: Take the last axis of d and its transpose (shape (3,))
    d_transpose = d[:, :, -1].T  # d_transpose shape: (3, 8)

    # Step 2: Compute the dot product with G (shape (200, 8))
    g_ = G[:, :, -1]
    dot_products = np.dot(g_, d_transpose)  # dot_products shape: (200, 3)

    # Step 3: Multiply each time step of G by the corresponding number from step 2
    multiplied_arrays = G * dot_products[..., np.newaxis]  # multiplied_arrays shape: (200, 8, 3)

    # Step 4: Add up these arrays along the third axis (shape (200, 8, 3))
    summed_array = np.sum(multiplied_arrays, axis=1)  # summed_array shape: (200, 3)

    # Step 5: Sum this final array along the first axis (shape (3,))
    final_result = np.sum(summed_array, axis=0)  # final_result shape: (3,)
    print("Fuck off")
