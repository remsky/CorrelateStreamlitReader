# %%
from warnings import resetwarnings
import spacy
import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from wordcloud import WordCloud
import pickle
import matplotlib.pyplot as plt

import streamlit as st
st.set_page_config(layout="wide")


from lib.data_classes import PubMedData
from lib.processer_classes import TransformForKG
#from lib.model_classes import ClusterModel


#%%

@st.cache
def load_abstract_data(search_term, quantity):
    df = PubMedData(search_term, quantity).df
    return df

@st.cache
def load_dataset(path):
    path_adjusted = os.path.join(os.path.dirname(__file__), path)
    with open(path_adjusted, 'rb') as handle:
        
        data = pickle.load(handle)
    return data

@st.cache(allow_output_mutation=True)
def load_en_ner_bionlp13cg_md():
    return spacy.load("en_ner_bionlp13cg_md")

# def load_en_core_sci_lg_linker(nlp):
#     if "scispacy_linker"in nlp.pipe_names:
#         return nlp
#     else:
#         return nlp.add_pipe("scispacy_linker", config={ "linker_name": "mesh"})

def load_transformed_data_KG(data,nlp):
    return TransformForKG(data,nlp)

# %%

# %%

st.title("Correlate AI: Abstract Reader")
with st.sidebar:
    with st.form("Search topic:"):
        preloaded = st.checkbox("Preloaded dataset (Ocular Disease)", value=False)
        user_search = st.text_input("Search for your academic topic", "")
        search_quantity = st.number_input('How any articles to return?',min_value=100,max_value=5000,value=200)
        submitted = st.form_submit_button("Correlate it!")

if (preloaded and submitted):
    st.write("Loading preloaded")
    with st.spinner('Dataset loading...'):
        data = load_dataset(r"ocular_data_5000.pkl")
elif ((not preloaded) and submitted):
    st.write("Loading live data")
    with st.spinner('Data downloading...'):
        data = load_abstract_data(user_search, search_quantity)

if (submitted):
    # # data.to_csv(r"C:\Users\jdavb\Desktop\CorrWeb\StreamlitExplorerMVP\test_data\ocular_data_3000.csv")
    loading_bar = st.progress(10)
    nlp = load_en_ner_bionlp13cg_md()
    loading_bar.progress(40)
    st.info("Knowledge graph engine loaded")
    data = data.head(100)
    increment = int(40/(len(data)))
    counter = 40
    for ind in range(0,len(data),3):
        counter += increment
        loading_bar.progress(counter)
        
        with st.expander(data.iloc[ind]['title']):
            try:
                processed_data = load_transformed_data_KG(data.iloc[ind],nlp)
                if processed_data.kg_strings == None:
                    continue
                
                col1, col2 = st.columns([5, 2])
                col1.subheader(data.iloc[ind]['title'])
                col1.write(data.iloc[ind]['abstract'])

                col2.header('Insights')
                for pair in processed_data.kg_strings:
                    col2.write(pair)
            except Exception:
                pass
    loading_bar.progress(100)
# # %%
# data = load_dataset(r"C:\Users\jdavb\Desktop\CorrWebTrends\StreamlitExplorerMVPTrends\ocular_data_5000.pkl")
# nlp = load_en_ner_bionlp13cg_md()
# #%%
# processed_data = load_transformed_data_KG(data.iloc[10],nlp)

# # %%

# # #%%
# # cluster_model = ClusterModel(processed_data.transformed_df)
# # # %%

# # processed_data.transformed_df.columns
# # # %%
# # pio.renderers.default = "notebook"
# # cluster_model.trend_fig.show()
