import matplotlib.pyplot as plt
import numpy as np
from preprocess_movielens_dataset import (get_posteriors_alphas_by_movie)
from sample_rankings_util import get_posteriors, get_mean_merits
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib import lines

def plot_all(fd_matrices, lp_matrices, alphas, cmap_name=None, figsize=(20, 40),
            save_at=None):
    
    fig = plt.figure(figsize=figsize)
    gs1 = gridspec.GridSpec(2, len(alphas))
    gs1.update(wspace=0.05, hspace=0.05) # set the spacing between axes. 
    num_docs = len(fd_matrices[alphas[0]][0])
    for i, alpha in enumerate(alphas):
        ax1 = plt.subplot(gs1[i])
        plt.axis('on')
        ax1.set_aspect('equal')       

        ax1.matshow(fd_matrices[alpha], vmin=0.0, vmax=1.0, cmap=plt.get_cmap(cmap_name) if cmap_name else None)
        ax1.set_title(r'$\phi={}$'.format(alpha),  y=1.35)
        if i == 0:
            ax1.set_ylabel('Movie')
            ax1.tick_params(labelbottom=False, labeltop=True, labelleft=True , labelright=False, labelsize=9)
            locs = ax1.yaxis.get_ticklocs()
            ax1.yaxis.set_ticklabels(map(lambda x: int(x)+1, locs))
        else:
            ax1.tick_params(labelbottom=False, labeltop=True, labelleft=False , labelright=False, labelsize=9)
        ax1.xaxis.set_label_position('bottom')
        ax1.yaxis.set_label_position('left')
        start, end = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(start, end, 10))
        locs = ax1.xaxis.get_ticklocs()
        ax1.xaxis.set_ticklabels(map(lambda x: int(x)+1, locs))
        
        
        ax2 = plt.subplot(gs1[i+len(alphas)])
        
        plt.axis('on')
        ax2.set_aspect('equal') 
        im = ax2.matshow(lp_matrices[alpha], vmin=0.0, vmax=1.0, cmap=plt.get_cmap(cmap_name) if cmap_name else None)
        ax2.set_xlabel('Position')
        start, end = ax2.get_xlim()
        ax2.xaxis.set_ticks(np.arange(start, end, 10))
        locs = ax2.xaxis.get_ticklocs()
        ax2.xaxis.set_ticklabels(map(lambda x: int(x)+1, locs))

        if i == 0:
            ax2.set_ylabel('Movie')
            ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True , labelright=False , labelsize=9)
            locs = ax2.yaxis.get_ticklocs()
            ax2.yaxis.set_ticklabels(map(lambda x: int(x)+1, locs))
        else:
            ax2.tick_params(labelbottom=True, labeltop=False, labelleft=False , labelright=False, labelsize=9 )
        ax2.xaxis.set_label_position('bottom')
        ax2.yaxis.set_label_position('left')
        
    if save_at:
        plt.savefig(save_at, 
                dpi=300, bbox_inches='tight', transparent=True)
    # fig.text(0.50, 0, '(c)', fontsize=18)
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title(r'$p^{\pi}_{m,k}$')
    
    fig.text(-0.02, 0.30, r'$\pi^{\small {\rm LP}, \phi}$', fontsize=16)
    fig.text(-0.02, 0.69, r'$\pi^{\small {\rm Mix}, \phi}$', fontsize=16)
        
    return fig

def plot_stacked_distributions(movie_ids,  posteriors_alpha_by_movie, rank_by_mean=True, sort=False,
                       rng=np.random.RandomState(), num_samples=1000, ax=None, savefig=False, util_transform=lambda x:x, **kwargs):
    # posteriors_alpha_by_movie, prior_alphas = get_posteriors_alphas_by_movie(probability_of_r, ratings_data, scaling_factor)
    means_dict = get_mean_merits(movie_ids, posteriors_alpha_by_movie, util_transform, return_dict=True)
    # print(means, movie_ids)
    if sort:
        movie_ids = sorted(movie_ids, key=lambda x: means_dict[x], reverse=True)
    means_arr = [means_dict[i] for i in movie_ids]
    
    posteriors = get_posteriors(movie_ids, posteriors_alpha_by_movie)
    blue, = sns.color_palette("muted", 1)
    fig, ax = plt.subplots(len(movie_ids), 1)
    fig.set_size_inches(7,len(movie_ids))
    for i in range(len(movie_ids)):
        exp_scores = []
        for j in range(num_samples):
            thetas_for_movie = rng.dirichlet(posteriors[i])
            if rank_by_mean:
                exp_score = np.sum([thetas_for_movie[i]*util_transform(i+1) for i in range(5)])
            else:
                exp_score = rng.choice(5, p=thetas_for_movie) + 1 + 1e-12 * rng.random()
                exp_score = util_transform(exp_score)
            exp_scores.append(exp_score) 
        sns.distplot(exp_scores, color=blue, ax=ax[i], hist_kws={'density':True},  **kwargs)
        ax[i].vlines(means_arr[i], 0.0, 1) #, labels=['Movie {}'.format(i+1) for i in range(len(means))])
        ax[i].set_xlim(1, 5)
        ax[i].set_ylim(0, 5)
    plt.tight_layout()
    if savefig:
        plt.savefig('paper/Uncertainity-based-fairness-for-rankings/figs/movielens_distribution_s_1e-4.pdf', 
                    dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def plot_distributions(movie_ids, posteriors_alpha_by_movie, rank_by_mean=True, 
                       rng=np.random.RandomState(), num_samples=1000, ax=None, util_transform=lambda x:x, title=None,
                       **kwargs):
    means = get_mean_merits(movie_ids, posteriors_alpha_by_movie, util_transform)
    posteriors = get_posteriors(movie_ids, posteriors_alpha_by_movie)

    blue, = sns.color_palette("pastel", 1)
    ax = plt.gca() if ax is None else ax
    for i in range(len(movie_ids)):
        exp_scores = []
        for j in range(num_samples):
            thetas_for_movie = rng.dirichlet(posteriors[i])
            if rank_by_mean:
                exp_score = np.sum([thetas_for_movie[i]*util_transform(i+1) for i in range(5)])
            else:
                exp_score = rng.choice(5, p=thetas_for_movie) + 1 
                exp_score = util_transform(exp_score) + 1e-12 * rng.random()
            exp_scores.append(exp_score) 
        sns.distplot(exp_scores, color=blue, ax=ax, **kwargs)
    ax.vlines(means, 0, 0.1,  linewidth=1.0, zorder=3)
    vertical_line = lines.Line2D([], [], color='black', marker='|', linestyle='None',
                          markersize=10, markeredgewidth=1.5, label=r'Expected Merits $\overline{v}_m$')
    ax.legend(handles=[vertical_line])
    ax.set_title(title)
    ax.set_xlim(1.0, 5.0)
    ax.set_ylim(0.0)
    ax.set_xticks([1, 2, 3, 4, 5])