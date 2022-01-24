import json
import time
import nltk
from nltk import tokenize

nltk.download('punkt')

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


def invoke_lfqa_model(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def query_audio_tts(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL_TTS, headers=headers, data=data)
    return response.content


def request_context_passages(question, header):
    response = requests.request("GET", CONTEXT_API_URL + question, headers=header)
    return json.loads(response.content.decode("utf-8"))


@st.cache(allow_output_mutation=True)
def get_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')


def signJWT() -> Dict[str, str]:
    payload = {
        "expires": time.time() + 6000
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def extract_sentences_from_passages(passages):
    sentences = []
    for idx, node in enumerate(passages):
        sentences.extend(tokenize.sent_tokenize(node["text"]))
    return sentences


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
            .row-widget.stTextInput > label {
                color: #b3b3b3;
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
                color: var(--text-color);
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
                color: var(--text-color);
            }
            #wikipedia-assistant {
                font-size: 36px;
            }
            .generated-answer {
                margin-bottom: 40px;
                border-left: 4px solid #ffc423;
                padding-left: 20px;
            }
            .generated-answer p {
                font-size: 16px;
                font-weight: bold;
            }
            .react-json-view {
                margin-bottom: 80px;
            }
        </style> """, unsafe_allow_html=True)

    footer = """
        <div class="footer-custom">
            Streamlit app created by <a href="https://www.linkedin.com/in/danijel-petkovic-573309144/" target="_blank">Danijel Petkovic</a>
        </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

    st.title('Wikipedia Assistant')

    question = st.text_input(
        label='Ask Wikipedia an open-ended question below; for example, "Why do airplanes leave contrails in the sky?"')
    if len(question) > 0:
        with st.spinner("Generating an answer..."):

            jwt_token = signJWT()
            header = {"Authorization": f"Bearer {jwt_token}"}
            context_passages = request_context_passages(question, header)

            conditioned_context = "<P> " + " <P> ".join([d["text"] for d in context_passages])
            model_input = f'question: {question} context: {conditioned_context}'

            data = invoke_lfqa_model({
                "inputs": model_input,
                "parameters": {
                    "truncation": "longest_first",
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

            st.markdown(
                " ".join([
                    "<div class='generated-answer'>",
                    f'<p>{generated_answer}</p>',
                    "</div>"
                ]),
                unsafe_allow_html=True
            )

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

            st.markdown("""<hr></hr>""", unsafe_allow_html=True)

            model = get_sentence_transformer()

            st.subheader("Context")
            selection = st.selectbox(label='Scope', options=('Paragraphs', 'Sentences'))

            question_e = model.encode(question, convert_to_tensor=True)
            if selection == "Paragraphs":
                context_e = model.encode([d["text"] for d in context_passages], convert_to_tensor=True)
                scores = util.cos_sim(question_e.repeat(context_e.shape[0], 1), context_e)
                similarity_scores = scores[0].squeeze().tolist()
                for idx, node in enumerate(context_passages):
                    node["answer_similarity"] = "{0:.2f}".format(similarity_scores[idx])
                context_passages = sorted(context_passages, key=lambda x: x["answer_similarity"], reverse=True)
                st.json(context_passages)
            else:
                sentences = extract_sentences_from_passages(context_passages)
                sentences_e = model.encode([sentence for sentence in sentences], convert_to_tensor=True)
                scores = util.cos_sim(question_e.repeat(sentences_e.shape[0], 1), sentences_e)
                sentence_similarity_scores = scores[0].squeeze().tolist()
                result = []
                for idx, sentence in enumerate(sentences):
                    result.append(
                        {"text": sentence, "answer_similarity": "{0:.2f}".format(sentence_similarity_scores[idx])})
                context_sentences = json.dumps(sorted(result, key=lambda x: x["answer_similarity"], reverse=True))
                st.json(context_sentences)

        else:
            unknown_error = f"{data}"
            st.warning(unknown_error)
