import streamlit as st
import YtAssistant as yt
import textwrap

st.title("Youtube Assistant")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.sidebar.text_area(
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
    response = yt.response_from_query(db, query)  # Corrected to one value
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))
