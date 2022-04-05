import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import *
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 

from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

pd.set_option("display.min_rows", 10)
pd.set_option("display.max_columns", 50)
pd.set_option("max_colwidth", 50)

import requests

def wikipedia_page(title):
    '''
    This function returns the raw text of a wikipedia page 
    given a wikipedia page title
    '''
    params = { 
        'action': 'query', 
        'format': 'json', # request json formatted content
        'titles': title, # title of the wikipedia page
        'prop': 'extracts', 
        'explaintext': True
    }
    # send a request to the wikipedia api 
    response = requests.get(
         'https://en.wikipedia.org/w/api.php',
         params= params
     ).json()

    # Parse the result
    page = next(iter(response['query']['pages'].values()))
    # return the page content 
    if 'extract' in page.keys():
        return page['extract']
    else:
        return "Page not found"


def filling_factor(df):
    '''
    Calculated the missing rate of a column
    '''
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['filling_factor'] = \
        (df.shape[0] - missing_df['missing_count']) / df.shape[0] * 100
    missing_df = \
        missing_df.sort_values('filling_factor').reset_index(drop=True)

    return missing_df

def barchart_percent(x, data, figsize=(12, 8), rotation=False):
    '''
    Plot barchart with plot
    '''
    plt.figure(figsize=figsize)
    ax = sns.countplot(x=x,
                       data=data)
    total = len(data)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%\n'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='center')
        
    if rotation == True:
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_n_gramms(data, use_tfidf:bool, ngram_ranges:list=[(1,1)], titles=['Unigramme BoW']):
    '''
    Sum each weihgt for each word
    '''

    if len(ngram_ranges) != len(titles):
        print('Please define as much titles as range')
        return

    else:
        
        fig = plt.figure(figsize=(18, len(ngram_ranges)*6))
        plt.suptitle('Weighted sum for each word', fontsize=24)
        
        for ind, ngram in enumerate(ngram_ranges):
            # Computing
            if use_tfidf == False:
                vec = CountVectorizer(ngram_range=ngram)
            else:
                vec = TfidfVectorizer(ngram_range=ngram)
    
            tf = vec.fit_transform(data)
            tmp = pd.DataFrame(tf.toarray(),columns=vec.get_feature_names_out()).sum().sort_values(ascending=False)[:10]
            
            # Displaying
            ax = fig.add_subplot(len(ngram_ranges), 1, ind+1)
            ax.set_title(titles[ind])
            sns.barplot(x=tmp.index,y=tmp.values, ax=ax, palette=sns.color_palette('deep'))

            total = len(tmp)
            for p in ax.patches:
                count = f'{p.get_height():0.0f}\n'
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                ax.annotate(count, (x, y), ha='center', va='center')

        plt.show()
    


def elbow_eval(X, model, k_min:int, k_max:int):
    '''
    Elbow method with differents metrics
    '''
    metrics = ["distortion", "silhouette", "calinski_harabasz"]
    i = 0

    fig, axes = plt.subplots(nrows=3,
                             ncols=1,
                             sharex=False,
                             sharey=False,
                             figsize=(10, 24))

    # Plot for each differents metrics
    for m in metrics:
        kmeans_visualizer = KElbowVisualizer(model,
                                             k=(k_min, k_max),
                                             metric=m,
                                             ax=axes[i])
        kmeans_visualizer.fit(X)
        kmeans_visualizer.finalize()
        i += 1

    plt.show()


def silouhette_eval(X, k_min:int, k_max:int):
    '''
    Silouhette score evaluation and visualisation
       & Intercluster distance Map with best k
    '''
    for k in range(k_min, k_max + 1):
        
        # Initialize the clusterer with k clusters value and a random generator
        # seed of 123 for reproducibility.
        
        model = KMeans(n_clusters=k,
                       max_iter=1000,
                       n_init=10,
                       random_state=123)

        cluster_labels = model.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters    
        silhouette_avg = silhouette_score(X, cluster_labels)

        fig, axes = plt.subplots(nrows=1,
                                ncols=2,
                                sharex=False,
                                sharey=False,
                                figsize=(20, 8))

        visualizer = SilhouetteVisualizer(model,
                                          ax=axes[0])
        distance_visualizer = InterclusterDistance(model,
                                                   ax=axes[1])
        visualizer.fit(X)
        distance_visualizer.fit(X)

        visualizer.finalize()
        distance_visualizer.show()
        
        print(
        "For n_clusters =",
        k,
        "The average silhouette_score is :",
        silhouette_avg, '\n'
        )

def vectorized_ngrams(data, vec, min_words:list, ngram:tuple, print_res:bool):
    '''
    Build vectorizer for the data, using min_words paraméters
    
    @param min_words: Minimal number of occurency of a word
    @param ngram: N-gram to use
    @param print: print the shape and the sparcity of the created vector
    '''
    
    vectorized_words_values = []
    vectorizers = []
    
    for c in min_words:
        tmp = vec(min_df=c, ngram_range=ngram)

        vectorized_words_values.append(tmp.fit_transform(data))
        vectorizers.append(tmp)
        
    if print_res:
        for ind in range(0, len(vectorized_words_values)):
            # Materialize the sparse data
            data_dense = vectorized_words_values[ind].todense()

            print(min_words[ind], end=' occurency -- ')
             # Compute Sparsicity = Percentage of Non-Zero cells
            print(f"(ngramm:{ngram}) shape: {vectorized_words_values[ind].shape}, sparcity: {((data_dense > 0).sum()/data_dense.size)*100:.3f} %")

    return vectorized_words_values, vectorizers



def plot_all_aris_score(vectorized_values_list: list, tfidf:bool, true_labels, metric, x_value, model, legends):
    '''
    Shows all ARI score on the same plot
    '''

    fig = go.Figure()

    for name_ind, data in enumerate(vectorized_values_list):
        score = []
        best_n_list = []
        
        try:
            # Calculated ARI for a vectorisation
            for ind in range(0,len(data)):
                tmp_score = []
                for rep in range(1, 11):
                    X = data[ind]
                    
                    if tfidf:

                        # Find best n component for dimension reduction
                        svd = TruncatedSVD(n_components = np.min(X.shape) - 1)
                        normalizer = Normalizer(copy=False)
                        lsa = make_pipeline(svd, normalizer)

                        lsa.fit_transform(X)
                        best_n = np.argmax(np.cumsum(svd.explained_variance_ratio_) > 0.99)

                        # Dimension reduction with n best component
                        svd = TruncatedSVD(n_components=best_n)
                        lsa = make_pipeline(svd, normalizer)

                        X_reduced = lsa.fit_transform(X)

                        model.fit(X_reduced)
                    
                    else:
                        
                        # Find best n component for dimension reduction
                        svd = TruncatedSVD(n_components = np.min(X.shape) - 1)

                        svd.fit_transform(X)
                        best_n = np.argmax(np.cumsum(svd.explained_variance_ratio_) > 0.99)

                        # Dimension reduction with n best component
                        svd = TruncatedSVD(n_components=best_n)

                        X_reduced = svd.fit_transform(X)

                        model.fit(X_reduced)

                    if metric == 'ARI':
                        tmp_score.append(adjusted_rand_score(true_labels, model.labels_))

                    elif metric == 'homogeneity':
                        tmp_score.append(homogeneity_score(true_labels, model.labels_))

                    elif metric == 'completness':
                        tmp_score.append(completness_score(true_labels, model.labels_))

                    elif metric == 'V-measure':
                        tmp_score.append(v_measure_score(true_labels, model.labels_))
                    else: 
                        print("metric is not well spell. Choose from: 'ARI', 'homogeneity', 'completness' and 'V-measure'")

                best_n_list.append(best_n)
                
                if metric == 'ARI':
                    score.append(np.mean(tmp_score))

                elif metric == 'homogeneity':
                    score.append(np.mean(tmp_score))

                elif metric == 'completness':
                    score.append(np.mean(tmp_score))

                elif metric == 'V-measure':
                    score.append(np.mean(tmp_score))
                else: 
                    print("metric is not well spell. Choose from: 'ARI', 'homogeneity', 'completness' and 'V-measure'")

        except:
            pass
        
        # Show score in a scatter plot
        fig.add_trace(go.Scatter(x=x_value,
                                 y=score,
                                 customdata=best_n_list,
                                 hovertemplate ="score:%{y}, best n:%{customdata}",
                    mode='lines+markers',
                    name=legends[name_ind]))
    fig.update_layout(title=f'{metric} score vs. word minimal occurency',
                      width=1000,
                      height=400,
                      xaxis_title="Minimal occurency",
                      yaxis_title= f'{metric} score',
                      hovermode='x unified')

    fig.show()