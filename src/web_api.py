import sys
from pathlib import Path

#we need to set up sys.path so Python can import the package "src"
#compute project root = folder above `src`
ROOT = Path(__file__).resolve().parent.parent

#ensure root is on sys.path so `import src` works
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

#now we can safely import from src
#we need to do it like that, otherwise we will get an error when calling streamlit
from src.batch_prediction_class import batch_prediction

# 4. Instantiate your prediction class
bp = batch_prediction()

st.title('Transfer Target Predictions')

#bp.player_lst => in the batch_prediction class we have created the variable player_lst, which we call in here
player = st.selectbox('Choose a player', bp.player_lst)

if st.button('Predict'):
    binary_df = bp.binary_batch_preds(player)
    multi_df = bp.multiclass_batch_predict(player)

    st.subheader('Binary (Top-League vs Non-Top League)')
    st.dataframe(binary_df)

    st.subheader('Multiclass (Target League Probabilities)')
    st.dataframe(multi_df)

#how to activate the website:
#streamlit run src/web_api.py
