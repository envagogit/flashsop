import streamlit as st
import moviepy.editor as mp
from st_clickable_images import clickable_images
import pandas as pd
from pytube import YouTube
import os
import requests
from time import sleep

upload_endpoint = "https://api.assemblyai.com/v2/upload"
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
start_time_conversion = 1000

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
    print("----> Printing post response json")
    print(transcript_response.json())

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


st.title("Autotorial")
# language = st.selectbox("Language:", ("🇺🇸 🇦🇺 🇬🇧", "🇪🇸", "🇩🇪", "🇫🇷", "🇳🇱", "🇮🇹", "🇵🇹"))
file = st.file_uploader("Upload a video file to generate a tutorial")


if file is not None:
    st.cache_data.clear()
    # Tabs
    tab1, tab2 = st.tabs(
        [
            "Content 🗒️",
            "JSON",
        ]
    )

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

    # UI
    # trial = results.json()["words"][3]["text"]
    prompt = st.chat_input("Any questions?")
    with tab2:
        st.write(results.json())
    with tab1:
        st.header("Summary", anchor="S0")
        with st.sidebar:
            st.write("[Summary](#S0)")

        st.write(results.json()["summary"])
        i = 0

        for part in results.json()["iab_categories_result"]["results"]:
            st.subheader("Part " + str(i + 1), anchor="P" + str(i + 1))

            st.video(
                file,
                start_time=results.json()["iab_categories_result"]["results"][i][
                    "timestamp"
                ]["start"]
                / start_time_conversion,
            )
            st.write(results.json()["iab_categories_result"]["results"][i]["text"])
            st.divider()
            with st.sidebar:
                st.write("[Part " + str(i + 1) + "](#P" + str(i + 1) + ")")
            i = i + 1
