import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
import seaborn as sns
from sklearn.model_selection import train_test_split

quant_cols=['num_critic_for_reviews', 'duration',
       'director_facebook_likes', 'actor_3_facebook_likes', 
       'actor_1_facebook_likes', 'gross', 'num_voted_users', 'cast_total_facebook_likes',
       'facenumber_in_poster', 'num_user_for_reviews',  'budget', 'title_year', 'actor_2_facebook_likes',
       'aspect_ratio', 'movie_facebook_likes','imdb_score']
    
qual_cols=['color','director_name','genres','actor_1_name','movie_title',
          'actor_2_name','plot_keywords','movie_imdb_link','actor_3_name','language','country','content_rating']

skewed = ['num_critic_for_reviews', 'duration',
       'director_facebook_likes', 'actor_3_facebook_likes', 
       'actor_1_facebook_likes', 'gross', 'num_voted_users', 'cast_total_facebook_likes',
       'facenumber_in_poster', 'num_user_for_reviews',  'budget', 'title_year', 'actor_2_facebook_likes',
       'aspect_ratio', 'movie_facebook_likes']

def show_missing_featues(movie_data):
    missing=movie_data.columns[movie_data.isnull().any()].tolist()
    print('Dataset size: ',movie_data.shape[0])
    print('Features that are having missing data(%):')
    return missing

def distribution(movie_data, dtype, transformed = False):
    """
    Visualization code for displaying distributions of features
    """
    
    # Create figure
    i=j=0        
    if dtype=='quant':
        if transformed==True:
            cols=skewed
        else:
            cols=quant_cols
            
        fig,axes=plt.subplots(nrows=6,ncols=3)
        for col in cols:
            movie_data[col].plot(kind='hist',ax=axes[i,j],figsize=(25,25),title=col,bins=30)
            if j==2:
                j=0
                i+=1
            else:
                j+=1
    else:
        cols=qual_cols
        fig,axes=plt.subplots(nrows=4,ncols=3)
        for col in cols:
            movie_data[col].value_counts().head(10).plot(kind='barh',ax=axes[i,j],figsize=(25,15),title=col)
            if j==2:
                j=0
                i+=1
            else:
                j+=1
            
    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Features", \
            fontsize = 16, y = 1.03)
    elif dtype=='quant':
        fig.suptitle("Skewed Distributions of Continuous Features", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Distributions of Categorical Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()

def bivariate(movie_data):
    pp = sns.pairplot(movie_data[quant_cols], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))
    fig = pp.fig 
    fig.subplots_adjust(top=0.93, wspace=0.3)
    t = fig.suptitle('Movie Attributes Pairwise Plots', fontsize=14)

def feature_plot(importances, X_train, y_train,n=5):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:n]]
    values = importances[indices][:n]

    # Creat the plot
    fig = plt.figure(figsize = (28,5))
    plt.title("Normalized Weights for First 15 Most Predictive Features", fontsize = 16)
    plt.bar(np.arange(n), values, width = 0.6, align="center", color = '#ff9999', label = "Feature Weight")
    plt.bar(np.arange(n) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#66b3ff', label = "Cumulative Feature Weight")
    plt.xticks(np.arange(n), columns)
    plt.xlim((-0.5, 15.5))
    plt.ylabel("Weight", fontsize = 10)
    plt.xlabel("Feature", fontsize = 10)
    
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show() 
    
def split_training_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    print ("Training set has {} samples.".format(X_train.shape[0]))
    print ("Testing set has {} samples.".format(X_test.shape[0]))
    return X_train, X_test, y_train, y_test 