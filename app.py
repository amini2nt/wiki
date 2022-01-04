import json
import streamlit as st
import requests
import time
from typing import Dict
import jwt

JWT_SECRET = st.secrets["api_secret"]
JWT_ALGORITHM = st.secrets["api_algorithm"]
INFERENCE_TOKEN = st.secrets["api_inference"]
CONTEXT_API_URL = st.secrets["api_context"]

headers = {"Authorization": f"Bearer {INFERENCE_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/vblagoje/bart_lfqa"
API_URL_TTS = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"

def query_eli_model(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

def query_audio_tts(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL_TTS, headers=headers, data=data)
    return response.content

def get_context(question, header):
    response = requests.request("GET", CONTEXT_API_URL + question , headers=header)
    return response

def signJWT(question: str) -> Dict[str, str]:
    payload = {
        "question": question,
        "expires": time.time() + 6000
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

st.set_page_config(
    page_title="AI assistant",
    initial_sidebar_state="expanded"
)

st.markdown(""" 
    <style> 
        .row-widget.stTextInput > div:first-of-type {
            background: #fff;
            display: flex;
            border: 1px solid #dfe1e5;
            box-shadow: none;
            border-radius: 24px;
            height: 50px;
            width: auto;
            padding: 10px;
            margin: 10px auto 30px;
        }
        .row-widget.stTextInput > div:first-of-type:hover,
        .row-widget.stTextInput > div:first-of-type:focus {
            box-shadow: 1px 1px 2px 1px rgba(0, 0, 0, 0.2);
        }
        .row-widget.stTextInput .st-bq {
            background-color: #fff;
        }
        .row-widget.stButton > button {
            border-radius: 24px;
            background-color: #B6C9B1;
            color: #fff;
            border: none;
            padding: 6px 20px;
            float: right;
            background-image: none;
        }
        .row-widget.stButton > button:hover {
            box-shadow: 1px 1px 2px 1px rgba(0, 0, 0, 0.2);
        }
        .row-widget.stButton > button:focus {
            border: none;
            color: #fff;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: white;
            color: black;
            max-width: 698px;
            color: rgba(38, 39, 48, 0.4);
            font-size: 14px;
            height: 50px;
        }
        .footer a {
            color: #000;
        }
    </style> """, unsafe_allow_html=True)

st.title('AI Assistant')

question = st.text_input('Enter a question')

footer="""
    <div class="footer">
        Streamlit app created by <a href="https://www.linkedin.com/in/danijel-petkovic-573309144/" target="_blank">Danijel Petkovic</a>
    </div>
"""
st.markdown(footer,unsafe_allow_html=True)

if question:
    with st.spinner("Generating an answer..."):

        jwt_token = signJWT(question)
        
        header = {"Authorization": f"Bearer {jwt_token}"}

        context = get_context(question, header)

        context_ready = (json.loads(context.content.decode("utf-8")))

        context_list = []

        for i in context_ready[:6]:
            context_list.append(i["text"])

        conditioned_context = "<P> " + " <P> ".join([d for d in context_list])

        model_input = f'question: {question} context: {conditioned_context}'

        if model_input:
            data = query_eli_model({
                "inputs": model_input, 
                "parameters": {
                    "min_length": 64,
                    "max_length": 256,
                    "do_sample": False, 
                    "early_stopping": True,
                    "num_beams": 8,
                    "temperature": 1.0,
                    "top_k": None,
                    "top_p": None,
                    "no_repeat_ngram_size": 3,
                    "num_return_sequences": 1
                }
            })
            
    if data and data[0]['generated_text']:
        generated_answer = data[0]['generated_text']

        st.markdown(
            " ".join([
                "<div style='margin-bottom: 50px;'>",
                '<div style="padding: 30px;background-color: #B6C9B1; border-radius: 10px;">',
                f'<p>{generated_answer}</p>',
                "</div>",
                "</div>"
            ]),
            unsafe_allow_html=True
        )

        with st.spinner("Generating an audio..."):
            audio_file = query_audio_tts({
                "inputs": generated_answer,
                "parameters": {
                    "vocoder_tag": "str_or_none(none)",
                    "threshold": 0.5,
                    "minlenratio": 0.0,
                    "maxlenratio": 10.0,
                    "use_att_constraint": False,
                    "backward_window": 1,
                    "forward_window": 3,
                    "speed_control_alpha": 1.0,
                    "noise_scale": 0.333,
                    "noise_scale_dur": 0.333
                }
            })

            with open("out.flac", "wb") as f:
                f.write(audio_file)
                
                st.audio("out.flac")
    else:
        st.markdown("""
            <div style="padding: 30px;background-color: #edd380; border-radius: 10px;">
                <p>Model is loading, please try again in a few moments...<p>
            </div>
        """,unsafe_allow_html=True
        )
