# YouTube Assistant with LangChain and OpenAI

This project is a YouTube Assistant that answers questions about YouTube videos based on their transcripts. The assistant extracts transcripts from YouTube, processes the text using LangChain's text-splitting and embedding techniques, and generates answers to user queries using OpenAI's GPT models.

## Features
- Extracts video transcripts from YouTube.
- Processes transcripts with LangChain's text processing utilities.
- Answers user questions based on video content using `gpt-3.5-turbo-instruct`.

## Requirements
- Python 3.7 or above
- `langchain`
- `streamlit`
- `openai`
- `faiss-cpu`

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/Colazzx/Youtube-Assistant.git
    ```

2. Navigate into the project directory:

    ```bash
    cd Youtube-Assistant
    ```

3. Create a virtual environment:

    ```bash
    python -m venv env
    ```

4. Activate the virtual environment:

    - On Windows:
      ```bash
      .\env\Scripts\activate
      ```
    - On MacOS/Linux:
      ```bash
      source env/bin/activate
      ```

5. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Add your OpenAI API key to **Streamlit Secrets** when deploying to Streamlit or use an `.env` file locally.

2. Run the app locally using Streamlit:

    ```bash
    streamlit run main.py
    ```

3. Input the YouTube video URL and your question in the Streamlit interface, and the assistant will provide answers based on the video transcript.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [OpenAI](https://openai.com)
