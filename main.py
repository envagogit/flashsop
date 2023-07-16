import streamlit as st
import moviepy.editor as mp
import pandas as pd
from pytube import YouTube
import os
import requests
from time import sleep
import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from datetime import timedelta
import openai


upload_endpoint = "https://api.assemblyai.com/v2/upload"
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
start_time_conversion = 1000
use_fake_prompt = False
llm_temperature = 0.0
temp_multiple_choice_qs = 0.3
temp_step_sequencer = 0.5
temp_tool_list = 0.3
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


def create_text_with_times(results):
    text_with_times = ""
    i = 0
    for word in results.json()["words"]:
        if i == 0:
            text_with_times = (
                text_with_times + "[" + str(word["start"]) + "] " + word["text"]
            )
        elif "." in results.json()["words"][i - 1]["text"]:
            text_with_times = (
                text_with_times + " [" + str(word["start"]) + "] " + word["text"]
            )
        else:
            text_with_times = text_with_times + " " + word["text"]
        i = i + 1
    return text_with_times


def create_process_table(sequenced_steps):
    split1 = sequenced_steps.split("[")
    aux_start_time = 0
    start_times = []
    steps = []
    durations = []
    i = 0
    for part in split1:
        if "]" in part:
            split_part = part.split("]")  # str(timedelta(seconds=sec))
            if i != 0:
                durations.append(ms_to_hhmmss(int(split_part[0]) - aux_start_time))
            start_times.append(ms_to_hhmmss(int(split_part[0])))
            if ":" in split_part[1]:
                steps.append(split_part[1].split(":")[1])
            else:
                steps.append(split_part[1])
            aux_start_time = int(split_part[0])
            i = i + 1
    durations.append("0:0:0")
    table = pd.DataFrame(
        {"Step": steps, "Start Time": start_times, "Duration": durations}
    )
    return table


def ms_to_hhmmss(ms):
    sec_tot = round(float(ms) / 1000)
    sec = int(((sec_tot / 60) - float(int(sec_tot / 60))) * 60)
    min_tot = float(int(sec_tot / 60))
    min = int((min_tot / 60 - float(int(min_tot / 60))) * 60)
    hour = int(min_tot / 60)
    print(ms)
    time = str(hour) + ":" + str(min) + ":" + str(sec)
    print(time)
    return time


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
    prompt_vector = []
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
        prompt_vector.append(
            "Part "
            + str(i + 1)
            + "\n \n"
            + results.json()["iab_categories_result"]["results"][i]["text"]
            + "\n \n"
        )
        i = i + 1

    return prompt, prompt_vector


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


# Simple LLM
# @st.cache_data
def simple_llm_run(prompt, temp):
    main_memory = ConversationBufferMemory(
        input_key="user_input", memory_key="chat_history"
    )
    llm = OpenAI(temperature=temp)
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
    # Run LLM
    with get_openai_callback() as cb:
        main_prompt_answer = main_prompt_chain.run("")
        print(cb)
    return main_prompt_answer


def format_simple_llm_questions(text):
    parenthesis = text.split(">")
    i = 0
    for part in parenthesis:
        if part != ">":
            if i == 0:
                # st.write(part[:-2])
                st.markdown(
                    "<h5 style='text-align: center;'>" + part[:-2] + "</h5>",
                    unsafe_allow_html=True,
                )
            else:
                if part == parenthesis[len(parenthesis) - 1]:
                    with st.expander(str(i) + ") " + part.partition("[")[0]):
                        st.write(part.partition("[")[2].replace("]", ""))

                else:
                    with st.expander(str(i) + ") " + part.partition("[")[0]):
                        st.write(part[:-2].partition("[")[2].replace("]", ""))

            i = i + 1


st.set_page_config(page_title="Easy SOP", page_icon="⚡")
st.markdown(
    "<h1 style='text-align: center;'>Easy ⚡ SOP</h1>",
    unsafe_allow_html=True,
)
# language = st.selectbox("Language:", ("🇺🇸 🇦🇺 🇬🇧", "🇪🇸", "🇩🇪", "🇫🇷", "🇳🇱", "🇮🇹", "🇵🇹"))

if "continue" not in st.session_state:
    st.session_state["continue"] = False

if not st.session_state["continue"]:
    st.write(
        "<h5 style='text-align: center;'>Upload a video where a process is explained by audio</h5>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.write(
        "<div style='text-align: center;'>✨Wait for the magic...✨</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.write(
        "<div style='text-align: center;'>An SOP will be created with the following sections:</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    file = None
    col1, col2, col3 = st.columns([29, 40, 20])
    with col2:
        st.write("🧑‍🎓 Tutorial")
        st.write("❓ Test")
        st.write("⚙️ Process Optimisation")
        st.write("🛠️ Tools and Resources")
        st.write("")
    col1, col2, col3 = st.columns([4, 3, 4])
    with col2:
        st.session_state["continue"] = st.button("Start Easy ⚡ SOP", type="primary")
# st.experimental_rerun()

if st.session_state["continue"]:
    file = st.file_uploader("Upload a video file to generate a tutorial")

if file is not None:
    # Save file
    if st.session_state.file != file:
        st.cache_data.clear()
        st.session_state.file = file

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Tutorial🧑‍🎓",
            "Test❓",
            "Process Optimisation⚙️",
            "Tools and Resources🛠️",
            "JSON💻",
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

            # LLMs
            # Main prompt
            audio_text, audio_text_vector = create_main_prompt(results=results)
            main_prompt = st.secrets["pre_prompt"] + audio_text

            # Init LLM
            main_prompt_chain = init_llm(main_prompt)

            # Run LLM
            main_prompt_answer = run_llm(
                prompt="",  # the prompt was introduced in the init_llm funct, this is used for adding user input
                use_fake_prompt=use_fake_prompt,
                fake_answer=st.secrets["fake_answer"],
            )
            # Question creation LLM

    with tab5:
        with st.expander("OpenAI Response:"):
            st.write(main_prompt_answer)  # BORRAR
    # Format LLM output
    titles_formatted, summaries_formatted = format_llm_output(
        llm_output=main_prompt_answer
    )

    # UI Generation
    prompt = st.chat_input("Any questions?")
    with tab5:
        with st.expander("Main Prompt"):
            st.write(main_prompt)
        with st.expander("AssemblyAI results"):
            st.write(results.json())
        with st.expander("audio_text_vector"):
            st.write(audio_text_vector)
    with tab1:
        st.header("Summary", anchor="S0")
        with st.sidebar:
            st.write("[Summary](#S0)")

        st.write(results.json()["summary"])
        st.divider()
        i = 0
    with tab5:
        with st.expander("audio_text"):
            st.write(audio_text)
    for part in titles_formatted:
        # Divide in Cols
        with tab1:
            col1, col2 = st.columns([5, 2])
            # Add Subheader
            with col1:
                st.subheader(titles_formatted[i], anchor="P" + str(i + 1))
            # Add Mini video
            with col2:
                st.video(
                    file,
                    start_time=results.json()["iab_categories_result"]["results"][i][
                        "timestamp"
                    ]["start"],
                )
            # Add summary
            st.write(summaries_formatted[i])
            # Add multiple choice question
            prompt_aux = (
                st.secrets["multiple_choice_header_prompt"]
                + results.json()["iab_categories_result"]["results"][i]["text"]
            )
        with tab2:
            if i < len(titles_formatted):
                mult_choice_question = simple_llm_run(
                    prompt_aux, temp_multiple_choice_qs
                )
                format_simple_llm_questions(mult_choice_question)
        with tab1:
            st.divider()
            with st.sidebar:
                st.write("[" + titles_formatted[i] + "](#P" + str(i + 1) + ")")
            i = i + 1
    with tab3:
        text_with_times = create_text_with_times(results)
        sequenced_steps = simple_llm_run(
            st.secrets["step_sequencer_prompt"] + text_with_times, temp_step_sequencer
        )
        process_table = create_process_table(sequenced_steps)
        st.write(process_table)
    with tab5:
        with st.expander("sequenced_steps"):
            st.write(sequenced_steps)

    tool_list = simple_llm_run(
        st.secrets["tools_prompt"] + audio_text + st.secrets["tools_prompt2"],
        temp_tool_list,
    )
    if "none" not in tool_list:
        tool_vector = tool_list.split("\n")
        with tab5:
            with st.expander("tool_vector"):
                st.write(tool_vector)
        with tab4:
            for tool in tool_vector:
                if ":" in tool:
                    with st.expander(tool.split(":")[0]):
                        st.write(tool.split(":")[1])
    else:
        with tab4:
            st.write("No tools or resources found")
