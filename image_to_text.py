import os
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import requests
import streamlit as st

# Load environment variables from .env file
load_dotenv("env/.env")
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# img text
def img_text(url):
    # Use a pipeline as a high-level helper
    img_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = img_to_text(url)[0]["generated_text"]
    return text

def generate_story(data):
    template = """
    You are a story teller:
    You can generate a short story based on a simple narrative, the story should be 50 words
    context:{data}
    story:
    """

    llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo")
    prompt = PromptTemplate(template=template, input_variables=["data"])

    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)

    story = story_llm.predict(data=data)
    return story

def text_to_speech(message):
    API_URL = "https://api-inference.huggingface.co/models/suno/bark"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('george.mp3', 'wb') as file:
        file.write(response.content)

def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="🤖")
    st.header("Turn an image into an story text")
    upload_file = st.file_uploader("Choose an image..", type="jpg")
    if upload_file is not None:
        print(upload_file)
        bytes_data = upload_file.getvalue()
        with open(upload_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(upload_file, caption="Uploaded image", use_column_width=True)

        scenario = img_text(upload_file.name)
        story = generate_story(scenario)
        text_to_speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        audio_file = open("george.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")


if __name__ == '__main__':
    main()