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
    # TODO.1 calculate scaled alpha
    # TODO.1.1 using recursion find out all the alpha tildes and store it somewhere

    # TODO.1.2 using each alpha tilde at timstep 'n' compute cn and store that also

    # TODO.2 calculate scaled beta using recursion and the cn array

    # TODO.3 calculate gamma

    # TODO.4 calculate spring

    return gamma_list, xi_list


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
    pi = find_new_pi(gamma_list, pi)

    # 2. Finding new mu
    phi["mu"] = find_new_mu(gamma_list, x_list)

    # 3. Finding new A
    A = find_new_A(xi_list, A)

    # 4. Finding new sigma
    phi["sigma"] = find_new_sigma(gamma_list, x_list, phi["mu"])

    return pi, A, phi


""" Functions for calculating each new parameter"""


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


#TODO.x wrong
def find_new_A(xi_list, A):
    '''
    Fill the Ajk values one by one

    @param xi_list:
    @param A:
    @return:
    '''

    '''
    TODO.x
    
    Numerator:
    we have spring which is 200,7,3,3 
    
    for each obs:
        obs_collector = []
        for each time window (of size 2):
            we have a 3x3...
            element wise sum across all of time ( i.e cell wise addition)
        we will have a 3X3 at this point in time ( add this to obs collector)
    obs collector will now have 200, 3,3 ( N, k ,k ) -> squish the 200 and get one final 3x3
    
    Denominator:
    take this final 3x3 and squish across column to get 3x1
    
    Now do a column wise division 3x3/ 3x1 (columnar division) -> we should still get a 3x3 which will be the new A (k x k)
    
    '''


    xis = np.array(xi_list)
    num_states = A.shape[0]

    # Find each Ajk value first and populate the A array
    for j in range(num_states):
        for k in range(num_states):
            # select the jth and kth values across all observations across all time steps
            j_k_selection = xis[:, :, j, k]

            # sum across time steps
            sum_across_timesteps = np.sum(j_k_selection, axis = 1)

            #sum across number of observations
            sum_across_observations = np.sum(sum_across_timesteps)

            #populate the value in the matrix
            A[j,k] = sum_across_observations

    # Normalize each Ajk value with the sum of all values in the A array
    A /= np.sum(A)

    return A


#TODO.x wrong
def find_new_sigma(gamma_list, x_list, mu):
    gammas = np.array(gamma_list)
    xs = np.array(x_list)

    '''
    TODO.x
    
    Numerator:
    obscollector = []
    for every obs:
        timecollector = []
        for every timestep n:
            z will be size 3
            x will be 1 value - mu of size 3 -> something of size 3 -> dot with its transpose to get 1 value (v)
            v x z -> output of size 3
            add to timecollector
        now timecollector wil have N, k -> squish this to get k
        add this to obs collector
    squish across obs collector  to get k ( this is the numerator)
    
    Denominator:
    200,8,3 (squish 8 i.e N)-> 200,3 (squish across obs)-> 3
    
    array of 3/ array of 3 -> sigma of 3
    
    '''

    # Find xn - muk -> (Obs, N) - (k) => (Obs, N, K)
    diff = xs[:, :, np.newaxis] - mu

    # Find the transpose of this (Obs, K, N)
    diff_transpose = np.transpose(diff, (0, 2, 1))

    # Multiply ( Obs, N, k) * (Obs, k, N) => (Obs, N, N)
    mul_diff = np.matmul(diff, diff_transpose)

    # In the numerator multiply gamma with this product (Obs, N, k ) * ( Obs, k, k) => (Obs, N,K)
    sigma_num = np.matmul(gammas, mul_diff)
    # for the numerator sum across axis 1 (N) first and then axis 0 (Obs) next => (k)
    sigma_num = np.sum(sigma_num, axis=1)
    sigma_num = np.sum(sigma_num, axis=0)

    # in the denominator do the similar thing
    sigma_denom = np.sum(gammas, axis=1)
    sigma_denom = np.sum(sigma_denom, axis=0)

    # find the ratio and return it
    sigma =  sigma_num / sigma_denom

    sigma /= np.sum(sigma)
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

