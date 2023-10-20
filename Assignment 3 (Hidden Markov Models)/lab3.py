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
    kmeans = KMeans(n_clusters=n_states.item(), random_state=seed).fit(x_cat[:, None])
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
    alphas_hat, c = Estep.calculate_scaled_alphas_and_c(x_list, pi, phi, A)

    # 2.  calculate scaled beta
    betas_hat = Estep.calculate_scaled_betas(x_list, phi, A, c)

    # 3. calculate gamma
    gamma_list = alphas_hat * betas_hat

    # 4 calculate xi_list
    xi_list = Estep.calculate_xi(x_list, alphas_hat, betas_hat, c, phi, A)

    return gamma_list, xi_list


class Estep:
    '''
    Functions related to calculating alpha

    '''

    @staticmethod
    def calculate_scaled_alphas_and_c(x_list, pi, phi, A):
        xs = np.array(x_list)  # Shape (O, N)
        mu = phi["mu"]  # Shape (K)
        sigma = phi["sigma"]  # Shape (K)

        O = xs.shape[0]
        N = xs.shape[1]
        K = mu.shape[0]

        alphas_hat = []  # should eventually be a shape of (O,N,K)
        c = []  # should eventually be a shape of (O, N)

        for o in range(O):  # for each observation
            # for each observation keep track of the alphas and c for this particular observation
            alphas_hat_for_obs = []
            c_for_obs = []

            # base case for this observation
            x0 = xs[o, 0]
            alpha_1 = Estep.calculate_alpha_1(x0, pi, mu, sigma)  # should be of shape (K)
            c_1 = np.sum(alpha_1)
            alpha_1_hat = alpha_1 / c_1

            # store the base case
            c_for_obs.append(c_1)
            alphas_hat_for_obs.append(alpha_1_hat)

            # using this base case perform tabulation to find alpha tilde
            for n in range(1, N):
                x_n = xs[o, n]

                # calculate alpha_tilde for the nth timestep
                alpha_n_tilde = Estep.calculate_alpha_tilde(x_n, alphas_hat_for_obs, A, mu, sigma)

                # Find alpha_hat_n & c_n using the previously calculated alpha_n_tilde
                alpha_n_hat, c_n = Estep.calculate_alpha_hat_and_c(alpha_n_tilde)

                # store
                c_for_obs.append(c_n)
                alphas_hat_for_obs.append(alpha_n_hat)

            # store in the arrays across observations
            c.append(c_for_obs)
            alphas_hat.append(alphas_hat_for_obs)

        # return the final scaled alphas and the constants
        return np.array(alphas_hat), np.array(c)

    @staticmethod
    def calculate_alpha_1(x0, pi, mu, sigma):
        K = mu.shape[0]
        alpha_1 = np.zeros((K))

        for k in range(K):
            normal_prob = Estep.calculate_prob_using_gausian(x0, mu[k], sigma[k])
            alpha_1[k] = pi[k] * normal_prob

        return alpha_1

    @staticmethod
    def calculate_alpha_tilde(x, alphas_hat, A, mu, sigma):
        # shape (K)
        prob_x_given_z = Estep.calculate_prob_x_given_z(x, mu, sigma)
        prev_alpha_hat = alphas_hat[-1]
        prob_z_curr_given_prev_z = A

        # multiplying column wise
        prev_alpha_hat = prev_alpha_hat[:, np.newaxis]
        product = prev_alpha_hat * prob_z_curr_given_prev_z

        # do sigma across axis=0 i.e. marginalize away z_prev
        product_marg_away_z_prev = np.sum(product, axis=0)
        alpha_tilde = prob_x_given_z * product_marg_away_z_prev

        return alpha_tilde

    @staticmethod
    def calculate_alpha_hat_and_c(alpha_n_tilde):
        c_n = np.sum(alpha_n_tilde)
        alpha_n_hat = alpha_n_tilde / c_n
        return alpha_n_hat, c_n

    '''
    Functions for calculating beta
    '''

    @staticmethod
    def calculate_scaled_betas(x_list, phi, A, c):
        xs = np.array(x_list)  # Shape (O, N)
        mu = phi["mu"]  # Shape (K)
        sigma = phi["sigma"]  # Shape (K)

        O = xs.shape[0]
        N = xs.shape[1]
        K = mu.shape[0]

        betas_hat = []

        for o in range(O):
            betas_hat_for_obs = [None] * N
            c_obs = c[o]

            # base case
            betas_hat_for_obs[-1] = np.array([1.0] * K)

            # Iterate in reverse!
            for n in range(N - 2, -1, -1):
                x_n_next = xs[o, n + 1]

                # Shape (K)
                beta_hat_next = betas_hat_for_obs[n + 1]

                # Shape (K)
                prob_x_n_next_given_z = Estep.calculate_prob_x_given_z(x_n_next, mu, sigma)

                # Shape (K, K)
                prob_z_n_next_given_z = A

                # Find beta_hat_n+1 * p(x+1 | zn+1)
                beta_times_prob_x_given_z = beta_hat_next * prob_x_n_next_given_z

                # getting it ready for row wise multiplication
                product = beta_times_prob_x_given_z * prob_z_n_next_given_z

                # marg away the column zn+1
                betas_tilde_n = np.sum(product, axis=1)
                betas_hat_n = betas_tilde_n / c_obs[n + 1]

                # assign it
                betas_hat_for_obs[n] = betas_hat_n

            # append the betas_hat computed for each obs to the total
            betas_hat.append(betas_hat_for_obs)

        return np.array(betas_hat)

    '''Function to calculate the spring (dont know what that greek letter is so calling it a spring)'''

    @staticmethod
    def calculate_xi(x_list, alphas_hat, betas_hat, c, phi, A):
        mu = phi["mu"]
        sigma = phi["sigma"]
        xs = np.array(x_list)

        O = alphas_hat.shape[0]
        N = c.shape[1]

        xi = []
        for o in range(O):
            xi_for_obs = []
            c_obs = c[o]
            alphas_hat_obs = alphas_hat[o]  # shape (N,K)
            betas_hat_obs = betas_hat[o]  # shape (N,K)

            # go till last but one!
            for n in range(1, N):
                x_n = xs[o, n]
                alpha_n_minus_1 = alphas_hat_obs[n-1]  # Shape (K)
                beta_n = betas_hat_obs[n]  # Shape (K)
                prob_z_n_given_prev_z = A

                prob_x_next_given_z = Estep.calculate_prob_x_given_z(x_n, mu, sigma)  # Shape (K)

                #make it a matrix here itself (3,1) x (1,3), because we are dealing with both zn-1 and zn
                product_alpha_with_prob_x_given_z = alpha_n_minus_1[:,None] * prob_x_next_given_z[None,:]

                # element wise matrix multiplication
                product_with_transition = product_alpha_with_prob_x_given_z * prob_z_n_given_prev_z

                # row wise product here
                xi_for_n_and_next = product_with_transition * beta_n  # Shape is (K, K)
                xi_for_n_and_next /= c_obs[n]
                xi_for_obs.append(xi_for_n_and_next)

            xi.append(xi_for_obs)  # xi is of shape (N-1,K,K)
        return np.array(xi)

    '''
    Common helper functions for both alpha and beta
    '''

    @staticmethod
    def calculate_prob_x_given_z(x, mu, sigma):
        K = mu.shape[0]
        prob = np.zeros((K))
        for k in range(K):
            prob[k] = Estep.calculate_prob_using_gausian(x, mu[k], sigma[k])
        return prob

    @staticmethod
    def calculate_prob_using_gausian(x, mu, sigma):
        z = (x - mu) / sigma
        exponent = -0.5 * np.square(z)
        coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
        return coefficient * np.exp(exponent)


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

    i = 0
    while True:
        print(f"i: {i}")
        gamma_list, xi_list = e_step(x_list, pi, A, phi)
        pi_new, A_new, phi_new = m_step(x_list, gamma_list, xi_list)

        has_pi_converged = np.all(np.isclose(pi, pi_new, threshold))
        has_A_converged = np.all(np.isclose(A, A_new, threshold))
        has_phi_mu_converged = np.all(np.isclose(phi["mu"], phi_new["mu"], threshold))
        has_phi_sigma_converged = np.all(np.isclose(phi["sigma"], phi_new["sigma"], threshold))

        pi, A, phi = pi_new, A_new, phi_new

        if i > 100:
            print("Exiting after 100 loops")
            break

        if has_pi_converged and has_A_converged and has_phi_mu_converged and has_phi_sigma_converged:
            print("Reached convergence!")
            break

        i += 1

    return pi, A, phi
