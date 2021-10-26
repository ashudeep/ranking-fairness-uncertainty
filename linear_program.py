import numpy as np
import itertools
from sample_rankings_util import (get_posteriors, get_mean_merits, sample_rankings,
                                  optimal_ranking, compute_marginal_rank_probabilities,
                                  plot_marginal_rank_distribution,
                                  _sample_ranking_bernoulli)
from evaluation import get_v, compute_constraint_probability


def get_linprog_coeffs_all_permutations(posteriors, alpha, v_vec=None, util_transform=lambda x: x):
    num_docs = len(posteriors)
    if v_vec is None:
        v_vec = get_v(num_docs)
    means = posteriors
    all_permutations = list(itertools.permutations(range(num_docs)))
    sampled_rankings = sample_rankings([], posteriors, 100000,
                                       sample_ranking_fn=_sample_ranking_bernoulli)
    # print(sampled_rankings)
    constraint_probabilities = compute_constraint_probability(sampled_rankings)
    # print(alpha, constraint_probabilities)
    rhs = - alpha * constraint_probabilities.flatten()
    utv = np.zeros((len(all_permutations)))
    for i, perm in enumerate(all_permutations):
        u_ = means[[j for j in perm]]
        utv[i] = - np.dot(util_transform(u_), v_vec)
    constraint_probability_lhs = np.zeros((num_docs, num_docs, len(all_permutations)))
    for i in range(num_docs):
        for k in range(num_docs):
            for j, perm in enumerate(all_permutations):
                if i in perm[:k + 1]:  # position of i in perm is <= k then put 1.
                    constraint_probability_lhs[i, k, j] = 1
    constraint_probability_lhs = - \
        constraint_probability_lhs.reshape(num_docs * num_docs, len(all_permutations))
    A_eq = np.ones((1, len(all_permutations)))
    b_eq = [[1]]
    bounds = (0, 1)
    return {'c': utv, 'A_ub': constraint_probability_lhs, 'b_ub': rhs,
            'A_eq': A_eq, 'b_eq': b_eq, 'bounds': bounds,
            'constraint_probabilities': constraint_probabilities}


def get_linprog_coeffs_movielens(movieids, posteriors_alpha_by_movie, v_vec,
                                 alpha=1.0, num_samples=50000,
                                 util_transform=lambda x: x):
    num_docs = len(movieids)
    posteriors = get_posteriors(movieids, posteriors_alpha_by_movie)
    means = get_mean_merits(movieids, posteriors_alpha_by_movie, util_transform)
    sampled_rankings = sample_rankings(
        movieids, posteriors, num_samples, util_transform=util_transform)
    #print(alpha, constraint_probabilities)
    utv = np.zeros((num_docs, num_docs))
    for i in range(num_docs):
        utv[i] = -1 * means[i] * np.array(v_vec)
    utv = utv.flatten()
    # plot_marginal_rank_distribution(sampled_rankings)
    constraint_probabilities = alpha * (compute_constraint_probability(sampled_rankings,
                                                                       plot_marginal_rank_distribution=True))
    # print(constraint_probabilities)
    constraints_ineq_lhs = []
    constraints_ineq_rhs = []
    for i in range(num_docs):
        for j in range(num_docs):
            temp_mat = np.zeros((num_docs, num_docs))
            for k in range(j + 1):
                temp_mat[i, k] = 1
            constraints_ineq_lhs.append(temp_mat.flatten())
            constraints_ineq_rhs.append(constraint_probabilities[i, j])

    constraints_eq_lhs = []
    constraints_eq_rhs = []
    for i in range(num_docs):
        temp_mat = np.zeros((num_docs, num_docs))
        for j in range(num_docs):
            temp_mat[i, j] = 1
        constraints_eq_lhs.append(temp_mat.flatten())
        constraints_eq_rhs.append(1)
        temp_mat = np.zeros((num_docs, num_docs))
        for j in range(num_docs):
            temp_mat[j, i] = 1
        constraints_eq_lhs.append(temp_mat.flatten())
        constraints_eq_rhs.append(1)
    bounds = (0, 1)
    return {
        'c': np.array(utv), 'A_ub': -1 * np.array(constraints_ineq_lhs),
        'b_ub': -1 * np.array(constraints_ineq_rhs),
        'A_eq': np.array(constraints_eq_lhs), 'b_eq': np.array(constraints_eq_rhs),
        'bounds': bounds, 'constraint_probabilities': constraint_probabilities
    }
