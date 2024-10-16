import streamlit as st
import YtAssistant as yt
import textwrap

st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.sidebar.text_input(
            label="What is the YouTube video URL?",
            max_chars=50
        )

        query = st.sidebar.text_area(
            label="Ask me about the video?",
            max_chars=50,
            key="query"
        )

        submit_button = st.form_submit_button(label="Submit")

if query and youtube_url:
    db = yt.create_db_from_youtube_link(youtube_url)
    if db is not None:
        response = yt.response_from_query(db, query)
        st.subheader("Answer:")
        st.text(textwrap.fill(response, width=80))
    else:
        st.error("There was an issue processing the video. Please check the URL or try again later.")
