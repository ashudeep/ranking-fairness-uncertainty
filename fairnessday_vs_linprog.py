import matplotlib.pyplot as plt
from scipy.optimize import linprog
import tqdm
import numpy as np
from sample_rankings_util import NOISE_MAGNITUDE
from evaluation import (get_v, get_mean_dcg, get_dcg)
from linear_program import get_linprog_coeffs_all_permutations

def sample_rankings(posteriors, num_samples=100):
    rankings = []
    for i in range(num_samples):
        rankings.append(sample_ranking_bernoulli(posteriors))
    return rankings

def sample_ranking_bernoulli(posteriors):
    if np.any(posteriors > 1.0):
        raise ValueError("Bernoulli Parameters should be between 0 and 1. They are: ", posteriors)
    scores = []
    for p in posteriors:
        scores.append(np.random.choice(2, p=[1-p, p]))
    # add very small noise to break ties
    scores += np.random.random(len(posteriors)) * NOISE_MAGNITUDE
    return np.argsort(-scores)

def optimal_ranking(means):
    noisy_means = means + np.random.random(len(means)) * NOISE_MAGNITUDE
    return np.argsort(-noisy_means)

def get_fairness_day_dcg(posteriors, alpha, v=None, num_days=100):
    v = get_v(len(posteriors)) if v is None else v
    dcg_non_fd = get_dcg(optimal_ranking(posteriors), posteriors, v)
    sampled_rankings = []
    sampled_rankings = sample_rankings(posteriors, num_days)
    dcg_fd = get_mean_dcg(sampled_rankings, posteriors, v)
    return alpha * (dcg_fd) + (1-alpha)*dcg_non_fd

def compare_fairness_day_with_linprog(num_docs, 
posteriors=None, num_alphas=5, v=None, alphas=None):
    if posteriors is None:
        posteriors = np.random.beta(2, 5, num_docs)
    else:
        posteriors = np.array(posteriors)
    linprog_dcgs = []
    fairness_day_dcgs = []
    if alphas is None:
        alphas = np.linspace(0, 1, num_alphas+1)
    coeffs = get_linprog_coeffs_all_permutations(posteriors, 1.0, v)
    for alpha in tqdm.tqdm(alphas):
        res = linprog(coeffs['c'], coeffs['A_ub'], alpha*coeffs['b_ub'], coeffs['A_eq'], coeffs['b_eq'], coeffs['bounds'])
        linprog_dcgs.append(-res.fun)
        fairness_day_dcgs.append(get_fairness_day_dcg(posteriors, alpha, v=v, num_days=10000))
    plt.plot(alphas, linprog_dcgs, label='Linear Program', marker='+', linestyle=':')
    plt.plot(alphas, fairness_day_dcgs, label='Fairness Day', marker='+', linestyle=':', color='darkorange')
    plt.plot([0,1], [fairness_day_dcgs[0], fairness_day_dcgs[-1]], label='Fairness Day (theoretical)', linestyle='--', color='darkorange')
    plt.legend()
    plt.ylabel('Utility')
    plt.xlabel(r'$\alpha$')
    plt.tight_layout()
    plt.savefig('fairness_day_vs_Lp.pdf')
    plt.show()
    
    return alphas, linprog_dcgs, fairness_day_dcgs