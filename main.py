import streamlit as st
from app import *

st.set_page_config(page_title="Google Palm Demo",page_icon=":bookmark_tabs:")

# st.image("logo.jpg",width=100)
st.title('Query Database using Gemini')

question = st.text_input("Type in your query:")
submit = st.button("Submit")

if question and submit:
    chain=get_db_chain()
    try:
        response = chain.run(question)
        #print(chain)
        st.header("Answer")
        st.write(response)
    except Exception as e:
        print("Error:", e)
