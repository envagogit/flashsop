import streamlit as st
import moviepy.editor as mp
from st_clickable_images import clickable_images
import pandas as pd
from pytube import YouTube
import os
import requests
from time import sleep
import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.callbacks import get_openai_callback
import sys

upload_endpoint = "https://api.assemblyai.com/v2/upload"
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
start_time_conversion = 1000
use_fake_prompt = False
llm_temperature = 0.0
apikey = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = apikey

if "file" not in st.session_state:
    st.session_state.file = None

headers = {"authorization": st.secrets["auth_key"], "content-type": "application/json"}

# Custom Functions
def save_uploadedfile(uploadedfile):
    with open(os.path.join(os.getcwd() + "/temp_files/", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join(os.getcwd() + "/temp_files/", "")


def extract_audio(video_path, savename):
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(savename)


# Old func
@st.cache_data
def save_audio(url):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download()
    base, ext = os.path.splitext(out_file)
    file_name = base + ".mp3"
    os.rename(out_file, file_name)
    return yt.title, file_name, yt.thumbnail_url


@st.cache_data
def upload_to_AssemblyAI(save_location):
    CHUNK_SIZE = 5242880

    def read_file(filename):
        with open(filename, "rb") as _file:
            while True:
                print("chunk uploaded")
                data = _file.read(CHUNK_SIZE)
                if not data:
                    break
                yield data

    upload_response = requests.post(
        upload_endpoint, headers=headers, data=read_file(save_location)
    )
    print(upload_response.json())

    audio_url = upload_response.json()["upload_url"]
    print("Uploaded to", audio_url)

    return audio_url


@st.cache_data
def start_analysis(audio_url):

    ## Start transcription job of audio file
    data = {
        "audio_url": audio_url,
        "iab_categories": True,
        "content_safety": True,
        "summarization": True,
        "summary_type": "bullets",
        "summary_model": "informative",
    }
    transcript_response = requests.post(transcript_endpoint, json=data, headers=headers)

    transcript_id = transcript_response.json()["id"]
    polling_endpoint = transcript_endpoint + "/" + transcript_id

    print("Transcribing at", polling_endpoint)
    return polling_endpoint


@st.cache_data
def get_analysis_results(polling_endpoint):

    status = "submitted"

    while True:
        print(status)
        polling_response = requests.get(polling_endpoint, headers=headers)
        status = polling_response.json()["status"]

        if status == "submitted" or status == "processing" or status == "queued":
            print("not ready yet")
            sleep(10)

        elif status == "completed":
            print("creating transcript")

            return polling_response

            break
        else:
            print("error")
            return False
            break


@st.cache_data
def open_video_file(file):
    video_file = open(file, "rb")
    return video_file.read()


def create_main_prompt(results):
    prompt = ""
    i = 0
    for part in results.json()["iab_categories_result"]["results"]:
        prompt = (
            prompt
            + "Part "
            + str(i + 1)
            + "\n \n"
            + results.json()["iab_categories_result"]["results"][i]["text"]
            + "\n \n"
        )
        i = i + 1
    return prompt


# LLM Funct
def run_llm(prompt, use_fake_prompt, fake_answer):
    if use_fake_prompt == True:
        main_prompt_answer = fake_answer
    else:
        with get_openai_callback() as cb:
            main_prompt_answer = main_prompt_chain.run(prompt)
            print(cb)
    return main_prompt_answer


@st.cache_data
def init_llm(prompt):
    main_memory = ConversationBufferMemory(
        input_key="user_input", memory_key="chat_history"
    )
    llm = OpenAI(temperature=llm_temperature)
    prompt_template = PromptTemplate(
        input_variables=["user_input"],
        template=prompt + "{user_input}",
    )
    main_prompt_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=False,
        output_key="main_prompt_answer",
        memory=main_memory,
    )
    return main_prompt_chain


def format_llm_output(llm_output):
    splitted_output = llm_output.split("[Part")
    j = 0
    titles_formatted = []
    summaries_formatted = []
    for part in splitted_output:
        if j != 0:
            title_format = part.replace(" " + str(j) + ":] ", str(j) + ". ")
            summary_format = title_format.replace("[Summary" + str(j) + ". ", "")
            titles_formatted.append(summary_format.partition("\n")[0])
            summaries_formatted.append(summary_format.partition("\n")[2])
        j = j + 1
    return titles_formatted, summaries_formatted


st.markdown(
    "<h1 style='text-align: center;'>Fla⚡h SOP</h1>",
    unsafe_allow_html=True,
)
# language = st.selectbox("Language:", ("🇺🇸 🇦🇺 🇬🇧", "🇪🇸", "🇩🇪", "🇫🇷", "🇳🇱", "🇮🇹", "🇵🇹"))

file = st.file_uploader("Upload a video file to generate a tutorial")

if file is not None:
    # Save file
    if st.session_state.file != file:
        st.cache_data.clear()
        st.session_state.file = file

    # Tabs
    tab1, tab2 = st.tabs(
        [
            "Content 🗒️",
            "JSON",
        ]
    )
    with tab1:
        with st.spinner("Wait for it..."):
            # Save dragged video file
            path = save_uploadedfile(file)

            # Name dragged audio file
            save_location = path + "audio_file_ext.mp3"

            # Save audio file
            extract_audio(path + file.name, save_location)

            # upload mp3 file to AssemblyAI
            audio_url = upload_to_AssemblyAI(save_location)

            # start analysis of the file
            polling_endpoint = start_analysis(audio_url)

            # receive the results
            results = get_analysis_results(polling_endpoint)

            # LLM
            # Main prompt
            main_prompt = st.secrets["pre_prompt"] + create_main_prompt(results=results)

            # Init LLM
            main_memory = ConversationBufferMemory(
                input_key="user_input", memory_key="chat_history"
            )
            llm = OpenAI(temperature=0.4)
            main_prompt_chain = init_llm(main_prompt)

            # Run LLM
            main_prompt_answer = run_llm(
                prompt="",  # the prompt was introduced in the init_llm funct, this is used for adding user input
                use_fake_prompt=use_fake_prompt,
                fake_answer=st.secrets["fake_answer"],
            )
    with tab2:
        with st.expander("OpenAI Response:"):
            st.write(main_prompt_answer)  # BORRAR
    # Format LLM output
    titles_formatted, summaries_formatted = format_llm_output(
        llm_output=main_prompt_answer
    )

    # UI Generation
    prompt = st.chat_input("Any questions?")
    with tab2:
        with st.expander("Main Prompt"):
            st.write(main_prompt)
        with st.expander("AssemblyAI results"):
            st.write(results.json())
    with tab1:
        st.header("Summary", anchor="S0")
        with st.sidebar:
            st.write("[Summary](#S0)")

        st.write(results.json()["summary"])
        st.divider()
        i = 0

        for part in titles_formatted:
            col1, col2 = st.columns([5, 2])
            with col1:
                st.subheader(titles_formatted[i], anchor="P" + str(i + 1))

            with col2:
                st.video(
                    file,
                    start_time=results.json()["iab_categories_result"]["results"][i][
                        "timestamp"
                    ]["start"],
                )
            st.write(summaries_formatted[i])
            st.divider()
            with st.sidebar:
                st.write("[" + titles_formatted[i] + "](#P" + str(i + 1) + ")")
            i = i + 1
