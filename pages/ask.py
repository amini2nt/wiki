import json
import time
from typing import Dict

import jwt
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer, util

JWT_SECRET = st.secrets["api_secret"]
JWT_ALGORITHM = st.secrets["api_algorithm"]
INFERENCE_TOKEN = st.secrets["api_inference"]
CONTEXT_API_URL = st.secrets["api_context"]

headers = {"Authorization": f"Bearer {INFERENCE_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/vblagoje/bart_lfqa"
API_URL_TTS = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan"


def query_eli_model(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def query_audio_tts(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL_TTS, headers=headers, data=data)
    return response.content


def get_context(question, header):
    response = requests.request("GET", CONTEXT_API_URL + question, headers=header)
    return response


@st.cache(allow_output_mutation=True)
def get_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')


def signJWT() -> Dict[str, str]:
    payload = {
        "expires": time.time() + 6000
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def truncate(sentence, word_count):
    sa = sentence.split()
    return " ".join(sa[:word_count])


def app():

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
            .footer-custom {
                position: fixed;
                bottom: 0;
                width: 100%;
                color: #83838366;
                max-width: 698px;
                font-size: 14px;
                height: 50px;
                padding: 10px 0;
                z-index: 50;
            }
            .main {
                padding: 20px;
            }
            footer {
                display: none !important;
            }
            .footer-custom a {
                color: #70707066;
            }
        </style> """, unsafe_allow_html=True)

    footer = """
        <div class="footer-custom">
            Streamlit app created by <a href="https://www.linkedin.com/in/danijel-petkovic-573309144/" target="_blank">Danijel Petkovic</a>
        </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

    st.title('LFQA Assistant')

    question = st.text_input(label='Ask a question')
    if len(question) > 0:
        with st.spinner("Generating an answer..."):

            jwt_token = signJWT()
            header = {"Authorization": f"Bearer {jwt_token}"}
            context = get_context(question, header)
            context_ready = json.loads(context.content.decode("utf-8"))
            context_list = []
            for i in context_ready:
                context_list.append(truncate(i["text"], 128))

            conditioned_context = "<P> " + " <P> ".join([d for d in context_list])
            model_input = f'question: {question} context: {conditioned_context}'
            if model_input:
                data = query_eli_model({
                    "inputs": model_input,
                    "parameters": {
                        "min_length": st.session_state["min_length"],
                        "max_length": st.session_state["max_length"],
                        "do_sample": st.session_state["do_sample"],
                        "early_stopping": st.session_state["early_stopping"],
                        "num_beams": st.session_state["num_beams"],
                        "temperature": st.session_state["temperature"],
                        "top_k": None,
                        "top_p": None,
                        "no_repeat_ngram_size": 3,
                        "num_return_sequences": 1
                    },
                    "options": {
                        "wait_for_model": True
                    }
                })
        if 'error' in data:
            st.warning("Seq2Seq model for answer generation is loading, please try again in a few moments...")

        elif data and len(data) > 0:
            generated_answer = data[0]['generated_text']

            st.write(generated_answer)
            st.write("")

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
                },
                "options": {
                    "wait_for_model": True
                }
            })

            if audio_file:
                with st.spinner("Generating an audio..."):
                    with open("out.flac", "wb") as f:
                        f.write(audio_file)

                        st.audio("out.flac")
            else:
                st.write('TTS model is loading')

            model = get_sentence_transformer()
            question_e = model.encode(question, convert_to_tensor=True)
            context_e = model.encode(context_list, convert_to_tensor=True)
            scores = util.cos_sim(question_e.repeat(context_e.shape[0], 1), context_e)
            similarity_scores = scores[0].squeeze().tolist()
            for idx, node in enumerate(context_ready):
                node["answer_similarity"] = "{0:.2f}".format(similarity_scores[idx])

            st.subheader("Context paragraphs:")
            st.json(context_ready)

        else:
            unknown_error = f"{data}"
            st.warning(unknown_error)
