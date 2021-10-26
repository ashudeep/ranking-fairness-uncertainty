import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# parser = argparse.ArgumentParser()
   
# parser.add_argument('-r', '--fraction', type=float, default=1.0, 
#     help="subsample rate of the ratings data.")

# args = parser.parse_args()
GENRES = [
      'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
      'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
      'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
  ]

def load_movielens_data(fraction, datadir='./ml-latest-small/'):
  ratings = pd.read_csv(datadir+'ratings.csv')
  ratings.rating = np.ceil(ratings.rating)

  value_counts = ratings.rating.value_counts().to_dict()
  probability_of_r = [value_counts[float(i+1)] for i in range(5)]
  probability_of_r = np.array(probability_of_r)/np.sum(probability_of_r)
  # plt.bar(np.arange(5)+1, probability_of_r)

  ratings = subsample_dataset(ratings, fraction=fraction)
  movies = pd.read_csv(datadir+'movies.csv')
  movies = movies.set_index('movieId')
  GENRE_MAP = {genre: idx for idx, genre in enumerate(GENRES)}
  num_ratings_by_movie = get_num_ratings_by_movie(ratings)
  total_num_ratings_by_movie = {m: sum(num_ratings_by_movie[m]) for m in num_ratings_by_movie}
  movies_with_more_than_1 = [m for m in num_ratings_by_movie if total_num_ratings_by_movie[m]>5]
  movies = movies[movies.index.isin(movies_with_more_than_1)]

  return probability_of_r, movies, ratings

def subsample_dataset(ratings, fraction, seed=None):
  return ratings.sample(frac=fraction, replace=False, random_state=seed)

def get_num_ratings_by_movie(ratings_data):
  num_ratings_by_movie = {}
  for movie in ratings_data.movieId.unique():
    value_counts =  ratings_data[ratings_data.movieId == movie].rating.value_counts().to_dict()
    num_ratings_by_movie[movie] = [value_counts.get(float(i),0) for i in range(1,6)]
  return num_ratings_by_movie

def get_posteriors_alphas_by_movie(probability_of_r, ratings_data, scaling_factor):
  num_ratings_by_movie = get_num_ratings_by_movie(ratings_data)
  prior_alphas = probability_of_r * scaling_factor
  posteriors_alpha_by_movie = {
      movieid: [prior_alphas[i]+num_ratings_by_movie[movieid][i] for i in range(5)]
      for movieid in num_ratings_by_movie
  }
  return posteriors_alpha_by_movie, prior_alphas

# def get_expected_ratings_from_posteriors(posteriors_alpha_by_movie,
# ratings_data):
#   exp_ratings_by_movie = {
#       movieid: np.sum([(i+1)*(posteriors_alpha_by_movie[movieid][i]/np.sum(posteriors_alpha_by_movie[movieid])) for i in range(5)])
#       +np.random.random()*1e-12 for movieid in ratings_data.movieId.unique()
#   }
#   return exp_ratings_by_movie

# movies['genre_vec'] = None
# movies['posteriors'] = None
# movies['count'] = None
# for id, row in movies.iterrows():
#     genres = [
#       GENRE_MAP.get(genre, 0)
#       for genre in row.genres.strip('').split('|')
#     ]
#     genre_vec = np.zeros(len(GENRES), dtype=np.int)
#     genre_vec[genres] = 1
#     movies.loc[id]['genre_vec'] = genre_vec
#     movies.loc[id]['count'] = num_ratings_by_movie.get(id, None)
#     movies.loc[id]['posteriors'] = posteriors_alpha_by_movie.get(id, None)

def get_movie_ids_by_genre_dict(movies_df, subsample_size=None):
  movie_ids_by_genre = {}
  rng = np.random.RandomState(seed=None)
  for genre in GENRES:
      movies_from_genre = movies_df[movies_df.genres.str.contains(genre)].index.tolist()
      if subsample_size is not None and len(movies_from_genre) >= subsample_size:
        movie_ids_by_genre[genre] = rng.choice(movies_from_genre, size=subsample_size, replace=False)
      else:
        movie_ids_by_genre[genre] = movies_from_genre
  return movie_ids_by_genre