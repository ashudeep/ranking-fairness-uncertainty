import numpy as np
from sample_rankings_util import (get_posteriors, get_mean_merits, sample_rankings,
                                  optimal_ranking, get_mean_merits,
                                  compute_marginal_rank_probabilities, plot_matrix)


def EXP_TRANSFORM(x): return 2.0**x - 1
def EXP_TRANSFORM_01(x): return 2.0**((x - 1.0) / 4.0) - 1


def compute_constraint_probability(sampled_rankings, plot_marginal_rank_distribution=False):
    marginal_rank_prob = compute_marginal_rank_probabilities(sampled_rankings)
    if plot_marginal_rank_distribution:
        print("Plotting the marginal rank distribution...")
        plot_matrix(marginal_rank_prob)
    # print(marginal_rank_prob)
    return np.cumsum(marginal_rank_prob, axis=1)


def get_v(num_docs, v_distr='expo', v_beta=1.1, v_num1=2, v_gamma=0.99):
    if v_distr == 'log':
        return [1.0 / np.log2(2 + i) for i in range(num_docs)]
    elif v_distr == 'binary':
        v = [0.0] * num_docs
        v[:v_num1] = [1.0] * v_num1
        return v
    elif v_distr == 'soft_binary':
        v = [0.0] * num_docs
        v[:v_num1] = [v_gamma**i for i in range(v_num1)]
        return v
    elif v_distr == 'poly':
        return [1.0 / (1 + i)**v_beta for i in range(num_docs)]
    elif v_distr == 'expo':
        return [1.0 / v_beta**i for i in range(num_docs)]
    elif v_distr == 'dcg_k':
        return [1.0 / np.log2(2 + i) if i < v_num1 else 0 for i in range(num_docs)]
    elif v_distr == 'expo_k':
        return [1.0 / v_beta**i if i < v_num1 else 0 for i in range(num_docs)]


def get_dcg(ranking, means, v):
    return np.dot(np.array(means)[ranking], v)


def run_fairness_day(movieids, alpha, posteriors_alpha_by_movie,
                     num_days=10000, util_transform=lambda x: x):
    expected_scores = get_mean_merits(movieids, posteriors_alpha_by_movie, util_transform)
    posteriors = get_posteriors(movieids, posteriors_alpha_by_movie)

    sampled_rankings = []
    for _ in range(num_days):
        is_fairness_day = np.random.random() < alpha
        if is_fairness_day:
            ranking = sample_rankings(movieids, posteriors, num_samples=1,
                                      util_transform=util_transform)[0]
        else:
            ranking = optimal_ranking(movieids, posteriors_alpha_by_movie,
                                      util_transform=util_transform)
        sampled_rankings.append(ranking)
    return sampled_rankings


def get_mean_dcg(sampled_rankings, means, v_vec):
    num_docs = len(means)
    dcgs = []
    for ranking in sampled_rankings:
        dcgs.append(get_dcg(ranking, means, v_vec))
    return np.mean(dcgs)


def compute_unfairness(movieids, matrix, v, num_samples=10000, constraint_probabilities=None):
    num_docs = len(movieids)
    if constraint_probabilities is None:
        sampled_rankings = sample_rankings(movieids, get_posteriors(movieids), num_samples)
        constraint_probabilities = compute_constraint_probability(sampled_rankings)
    cumsum_matrix = np.cumsum(matrix, axis=1)
    diff_mat = np.clip(constraint_probabilities - cumsum_matrix, 0, np.inf)
    sum_diff_mat = np.sum(np.matmul(diff_mat, v))
    return sum_diff_mat / num_docs**2
