# Fairness in Ranking under Uncertainty

This repository is the official implementation of the NeurIPS 2021 paper [Fairness in Ranking under Uncertainty](https://arxiv.org/abs/2107.06720). 

<img src="neurips2021_thumbnail.png" alt="uncertainty" width="300"/>
<img src="https://camo.githubusercontent.com/b1dff6a6513fce2ebb171af6d4c6e446b6552dadfd6f15f9f71d0d8b1c8b7e26/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f656e2f7468756d622f302f30382f4c6f676f5f666f725f436f6e666572656e63655f6f6e5f4e657572616c5f496e666f726d6174696f6e5f50726f63657373696e675f53797374656d732e7376672f3132303070782d4c6f676f5f666f725f436f6e666572656e63655f6f6e5f4e657572616c5f496e666f726d6174696f6e5f50726f63657373696e675f53797374656d732e7376672e706e67" alt="neurips" width="300"/>

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset

In the paper, we present two sets of experiments -- a simulation based on the Movielens dataset, and a real-world paper recommender system run at KDD 2020.

The dataset may be downloaded from the [MovieLens website](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip), and unzipped in the subdirectory 'ml-latest-small'.

The details of the KDD recommender system experiment are shared in the kdd-experiment subdirectory.

## Running the experiments

To run the experiments, simply follow the steps as shown in the jupyter notebook 'movielens-ranking-fairness-uncertainty.ipynb'.

The notebook uses the library which contains the following files:

- preprocess_movielens_dataset.py: Provides functionality to load and preprocess the movielens dataset as described in the paper.
- linear_program.py: Provides necessary functions to transform the ranking problem to the linear program that can be solved by an off-the-shelf LP solver.
- sample_rankings.py: Provides utility functions to work with ranking distributions.
- plot_utils.py: Utility functions to plot the graphs and visualizations shown below.
-requirements.txt: To be able to run the code, these requirements must be installed. The jupyter notebook also requires a latex installation to be able to use math fonts inside the plots. If it does not work for you, please change the line with plt.rcParams['text.usetex'] from True to False.

Jupyter Notebooks:

- suboptimal-OPT-TS-mix.ipynb: Notebook with the example showing how OPT/TS mixing ranking policy can be suboptimal in terms of utility while the LP based ranking policy achieves optimal utility.
- movielens-ranking-fairness-uncertainty.ipynb: Notebook with the movielens experiments.

## Contributing

This project is licensed under the terms of the MIT license.
