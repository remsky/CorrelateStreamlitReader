# %%
from ast import keyword
import re
import json
import requests
import traceback
import numpy as np
import pandas as pd
from pymed import PubMed
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import OPTICS, cluster_optics_dbscan, SpectralClustering
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# %%

class ClusterModel:
    '''
    Not working great with current vectorization, rarely finds clusters
    '''
    def __init__(self,transformed_df,prepped_df=None):
        trend_df = transformed_df.copy()#[pd.to_datetime(transformed_df.publication_date) > (pd.Timestamp.today() - pd.Timedelta('30D'))]
        self.transformed_df = trend_df
        
        self.vectors = self.transformed_df.vectors
        self.title = self.transformed_df.title
        self.dates = self.transformed_df.publication_date
        self.fig = None
        self.trend_fig = None
        self.row_trend_fig = None
        self.prepped_df = prepped_df
        if prepped_df == None:
            self.run_spectral_clustering()
            self.plot_clusters()
            self.prep_plot_cluster_trends()
        self.plot_combined_cluster_trend(self.prepped_df)
        self.plot_separate_cluster_trend(self.prepped_df)

    def run_optics_clustering(self):
        '''Not great, rarely finds more than one cluster
        '''
        clust = OPTICS(min_samples=5, xi=0.01, min_cluster_size=0.05)
        a = np.array(self.vectors.to_list())
        clust.fit(a)
        self.transformed_df['clusters'] = clust.labels_[clust.ordering_]

    def run_spectral_clustering(self):
        sc = SpectralClustering(5, affinity='rbf', n_init=50,
                        assign_labels='kmeans')
        labels = sc.fit_predict(np.array(self.vectors.to_list()))  
        self.transformed_df['clusters'] = labels

    def rename_clusters(self):
        self.transformed_df['clusters'] = self.transformed_df['clusters'].astype(str)
        original_clusters = list(self.transformed_df.clusters.unique())
        for cluster in original_clusters:
            temp_df = self.transformed_df[self.transformed_df.clusters == cluster]
            keywords = temp_df.entities.value_counts().index.tolist()[0][:4]
            keyword_string = ", ".join(keywords)

            self.transformed_df['clusters'] = self.transformed_df['clusters'].str.replace(
                                                    str(cluster),keyword_string)

    def plot_clusters(self):
        pca = PCA(n_components=2)
        res = pca.fit_transform(np.array(self.vectors.to_list()))
        self.transformed_df['x_vector'],self.transformed_df['y_vector'] = np.hsplit(res,2)
        self.rename_clusters()
        fig = px.scatter(self.transformed_df, 
                    x="x_vector", 
                    y="y_vector",
                    color="clusters",
                    hover_data=['title'])
             #    size='petal_length' # number of authors? citations?

        fig.update_layout(showlegend=False,hovermode=None)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        
        self.fig = fig
        
    
    def prep_plot_cluster_trends(self):
        trend_df = self.transformed_df
        trend_df = trend_df[pd.to_datetime(trend_df.publication_date) > (pd.Timestamp.today() - pd.Timedelta('90D'))]
        #  and trend_df.publication_date < pd.Timestamp.today()]

        trend_df['publication_date'] = pd.to_datetime(trend_df['publication_date'])
        trend_df = trend_df.sort_values(by='publication_date')
        # trend_df['rolling']  = trend_df.set_index('publication_date')\
        #         .groupby('clusters', sort=False)['clusters']\
        #         .rolling('7d',min_periods=1).count()
        trend_df['rolling'] = trend_df.groupby('clusters')['publication_date']\
                                            .rolling(10,min_periods=1).count().reset_index(0,drop=True)

        trend_df['smoothed_rolling'] = trend_df.groupby('clusters')['rolling']\
                                            .rolling(2,min_periods=1,win_type='gaussian').mean(std=0.5).reset_index(0,drop=True)
        # fig = px.area(trend_df, x="publication_date", y="rolling", line_group='clusters',color='clusters')

        # self.trend_fig = fig
        self.prepped_df = trend_df.filter(['publication_date','smoothed_rolling','clusters'],axis=1)
        
    def plot_combined_cluster_trend(self,prepped_trend_df):
        fig = go.Figure()
        colorval = 0
        for clust in prepped_trend_df.clusters.unique():
            temp_df = prepped_trend_df[prepped_trend_df.clusters == clust]
            fig.add_trace(go.Scatter(
                x=temp_df['publication_date'],y=temp_df['smoothed_rolling'],
                hoverinfo='x+y',
                name=clust,
                mode='lines',
                line=dict(width=0.5, color=f'rgb(131, {90+colorval}, {241-colorval})'),
                stackgroup='one',
                groupnorm='percent',

            ))
            colorval += 30
        fig.update_xaxes(rangeslider_visible=True)
        self.trend_fig = fig
    
    def plot_separate_cluster_trend(self,prepped_trend_df):
        clusts = prepped_trend_df.clusters.unique()
        row_i = 1
        fig = make_subplots(rows=len(clusts), cols=1, shared_xaxes=True,shared_yaxes=True,vertical_spacing=0.02)
        for clust in clusts:
            temp_df = prepped_trend_df[prepped_trend_df.clusters == clust]
            fig.add_trace(go.Scatter(x=temp_df['publication_date'],y=temp_df['smoothed_rolling'],name=clust),
                        row=row_i, col=1)

            row_i += 1
        
        
        self.row_trend_fig = fig