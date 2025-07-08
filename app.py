import streamlit as st

# create a multi-page app

page_1 = st.Page('pages/home.py', title='Home', icon='🏠')
page_2 = st.Page('pages/main.py', title='PK Prediction', icon='🍀')
page_3 = st.Page('pages/documentation.py', title='Documentation', icon='📜')

pg = st.navigation([page_1, page_2, page_3])
pg.run()
