import streamlit as st
import pandas as pd
import os
import time
import numpy as np

episode = st.sidebar.selectbox("Episode of CSI", ['s01e07', 's01e08', 's02e01', 's02e04'])

dict_scores = {"s01e07" : 0.916,
"s01e08" : 0.919,
"s01e19" : 0.579,
"s01e20" : 0.746,
"s01e23" : 0.686,
"s02e01" : 0.880,
"s02e04" : 0.894,
"s02e06" : 0.855}

st.header("Graph2Speak: Visualizing the created networks")

os.system('cp src/generated_graph/%s/pred.html ../anaconda3/lib/python3.7/site-packages/streamlit/static/pred.html'%episode)
os.system('cp src/generated_graph/%s/truth.html ../anaconda3/lib/python3.7/site-packages/streamlit/static/truth.html'%episode)
os.system('cp src/generated_graph/%s/rerank.html ../anaconda3/lib/python3.7/site-packages/streamlit/static/rerank.html'%episode)
time.sleep(2)

st.subheader("Ground truth graph")
st.markdown('<iframe src="/truth.html" style="width: 700px; height: 550px; border: 0px"> </iframe>', unsafe_allow_html=True)

st.subheader("Predicted from SID")
st.write("Speaker accuracy: %s"%(np.round(dict_scores[episode] * 100, 2)))

st.markdown('<iframe src="/pred.html" style="width: 700px; height: 550px; border: 0px"> </iframe>', unsafe_allow_html=True)

st.subheader("Graph2Speak")
st.markdown('<iframe src="/rerank.html" style="width: 700px; height: 550px; border: 0px"> </iframe>', unsafe_allow_html=True)

st.subheader("Differences")
df = pd.read_csv("src/graph2speak_output/%s/diff.csv"%(episode), index_col=0)
for val in df.iterrows():
    st.write(val[1])