import numpy as np
import matplotlib.pyplot as plt

NOISE_MAGNITUDE = 0

def get_posteriors(movieids, posteriors_alpha_by_movie):
    return [posteriors_alpha_by_movie[mid] for mid in movieids]

def get_multinomial_expected_value(thetas, util_transform):
    return np.sum([thetas[i]*util_transform(i+1) for i in range(5)])

def get_mean_merits(movieids, posteriors_alpha_by_movie, util_transform, return_dict=False):
    exp_merits_by_movie = {
        movieid: np.sum([util_transform(i+1)*(
            posteriors_alpha_by_movie[movieid][i]/np.sum(posteriors_alpha_by_movie[movieid])) for i in range(5)])
        +np.random.random()*NOISE_MAGNITUDE for movieid in movieids
    }
    if return_dict:
        return exp_merits_by_movie
    else:
        exp_merits = np.array([exp_merits_by_movie[movieid] for movieid in movieids])
        return exp_merits

def _sample_ranking(movie_ids, posteriors, rank_by_mean=True, util_transform=None, 
        rng=np.random.RandomState()):
    exp_scores = []
    for i in range(len(movie_ids)):
        thetas_for_movie = rng.dirichlet(posteriors[i])
        if rank_by_mean:
            exp_score = get_multinomial_expected_value(thetas_for_movie, util_transform) + NOISE_MAGNITUDE * rng.random()
        else:
            exp_score = rng.choice(5, p=thetas_for_movie) + 1 
            exp_score = util_transform(exp_score) + NOISE_MAGNITUDE * rng.random()
        exp_scores.append(exp_score) 
    return np.argsort(-1*np.array(exp_scores))


def _sample_ranking_bernoulli(bernoulli_posteriors):
    if np.any(bernoulli_posteriors > 1.0):
        raise ValueError("Bernoulli Parameters should be between 0 and 1. They are: ", bernoulli_posteriors)
    scores = []
    for p in bernoulli_posteriors:
        scores.append(np.random.choice(2, p=[1-p, p]))
    # add very small noise to break ties
    scores += np.random.random(len(bernoulli_posteriors)) * 1e-10
    return np.argsort(-scores)

def sample_rankings(movie_ids, posteriors, num_samples=10, rank_by_mean=True, 
sample_ranking_fn=_sample_ranking, util_transform=lambda x: x):
    rankings = []
    for _ in range(num_samples):
        if sample_ranking_fn == _sample_ranking:
            rankings.append(_sample_ranking(movie_ids, posteriors, rank_by_mean, 
            util_transform=util_transform))
        elif sample_ranking_fn == _sample_ranking_bernoulli:
            rankings.append(_sample_ranking_bernoulli(posteriors))
        else:
            ValueError("Wrong input for sample ranking function.")
    return rankings


def optimal_ranking(movie_ids, posteriors_alpha_by_movie, util_transform=lambda x: x):
    expected_scores = get_mean_merits(movie_ids, posteriors_alpha_by_movie, util_transform)
    return np.argsort(np.array(expected_scores))[::-1]

def compute_marginal_rank_probabilities(sampled_rankings):
    num_rankings = len(sampled_rankings)
    num_docs = len(sampled_rankings[0])
    prob_matrix = np.zeros((num_docs, num_docs))
    for ranking in sampled_rankings:
        for i in range(num_docs):
            prob_matrix[ranking[i], i] += 1
    # print(prob_matrix)
    return prob_matrix / num_rankings

def plot_marginal_rank_distribution(sampled_rankings, genre=None):
  plot_matrix(compute_marginal_rank_probabilities(sampled_rankings), title=genre)

def plot_matrix(matrix, title=None):
  plt.matshow(matrix)
  if title:
    plt.title(title)
  plt.xlabel('Position')
  plt.ylabel('Item Id')
  plt.colorbar()
  plt.show()