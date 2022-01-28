import colorsys
import json
import time

import nltk
from nltk import tokenize

nltk.download('punkt')
from google.oauth2 import service_account
from google.cloud import texttospeech

from typing import Dict

import jwt
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer, util

JWT_SECRET = st.secrets["api_secret"]
JWT_ALGORITHM = st.secrets["api_algorithm"]
INFERENCE_TOKEN = st.secrets["api_inference"]
CONTEXT_API_URL = st.secrets["api_context"]
LFQA_API_URL = st.secrets["api_lfqa"]

headers = {"Authorization": f"Bearer {INFERENCE_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/vblagoje/bart_lfqa"
API_URL_TTS = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan"


def api_inference_lfqa(model_input: str):
    payload = {
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
    }
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def inference_lfqa(model_input: str, header: dict):
    payload = {
        "model_input": model_input,
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
        }
    }
    data = json.dumps(payload)
    response = requests.request("POST", LFQA_API_URL, headers=header, data=data)
    return json.loads(response.content.decode("utf-8"))


def hf_tts(text: str):
    payload = {
        "inputs": text,
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
    }
    data = json.dumps(payload)
    response = requests.request("POST", API_URL_TTS, headers=headers, data=data)
    return response.content


def google_tts(text: str, private_key_id: str, private_key: str, client_email: str):
    config = {
        "private_key_id": private_key_id,
        "private_key": f"-----BEGIN PRIVATE KEY-----\n{private_key}\n-----END PRIVATE KEY-----\n",
        "client_email": client_email,
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    credentials = service_account.Credentials.from_service_account_info(config)
    client = texttospeech.TextToSpeechClient(credentials=credentials)

    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(language_code="en-US",
                                              ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return response


def request_context_passages(question, header):
    response = requests.request("GET", CONTEXT_API_URL + question, headers=header)
    return json.loads(response.content.decode("utf-8"))


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_sentence_transformer_encoding(sentences):
    model = get_sentence_transformer()
    return model.encode([sentence for sentence in sentences], convert_to_tensor=True)


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


def similarity_color_picker(sentence_similarity: float):    
    value = int(sentence_similarity * 100)
    rgb = colorsys.hsv_to_rgb(value / 300., 1.0, 1.0)
    return [round(255 * x) for x in rgb]

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % tuple(rgb)


def similiarity_to_hex(sentence_similarity: float):
    return rgb_to_hex(similarity_color_picker(sentence_similarity))


def answer_to_context_similarity(generated_answer, context_passages, topk=3):
    context_sentences = extract_sentences_from_passages(context_passages)
    context_sentences_e = get_sentence_transformer_encoding(context_sentences)
    answer_sentences = tokenize.sent_tokenize(generated_answer)
    answer_sentences_e = get_sentence_transformer_encoding(answer_sentences)
    search_result = util.semantic_search(answer_sentences_e, context_sentences_e, top_k=topk)
    result = []
    for idx, r in enumerate(search_result):
        context = []
        for idx_c in range(topk):
            context.append({"source": context_sentences[r[idx_c]["corpus_id"]], "score": r[idx_c]["score"]})
        result.append({"answer": answer_sentences[idx], "context": context})
    return result


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
                margin: 40px 0 80px;
            }
            .row-widget.stSelectbox label {
                display: none;
            }
            .score-circle {
                width: 20px;
                height: 20px;
                border-radius: 50%;
                padding: 2px;
                font-size: 8px;
            }
            .tooltip {
                text-align: center;
                line-height: 20px;
                display: inline-block;
                font-size: 8px;
                border-radius: 50%;
                height: 20px;
                width: 20px;
                position: absolute;
                cursor: pointer;
            }

            .tooltip .tooltiptext {
                visibility: hidden;
                width: 120px;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;

                /* Position the tooltip */
                position: absolute;
                z-index: 1;
            }

            .tooltip:hover .tooltiptext {
                visibility: visible;
            }
            .tooltip .tooltiptext {
                width: 200px;
                bottom: 100%;
                left: 50%;
                margin-left: -100px;
                color: #000;
                font-size: 12px;
            }
            .stAudio {
                margin-top: 50px;
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

            if st.session_state["api_lfqa_selector"] == "HuggingFace":
                data = api_inference_lfqa(model_input)
            else:
                data = inference_lfqa(model_input, header)
        if 'error' in data:
            st.warning("Seq2Seq model for answer generation is loading, please try again in a few moments...")

        elif data and len(data) > 0:
            generated_answer = data[0]['generated_text']
            sentence_similarity = answer_to_context_similarity(generated_answer, context_passages, topk=3)
            sentences = "<div>"
            for item in sentence_similarity:
                sentences += '<span>'
                score = item["context"][0]["score"]
                formatted_score = "{0:.2f}".format(score)
                sentences += "".join([                    
                        f'{item["answer"]}',
                        f'<span style="background-color: #{similiarity_to_hex(score)}" class="tooltip">',
                            f'{formatted_score}',
                            f'<span style="background-color: #{similiarity_to_hex(score)}" class="tooltiptext">{item["context"][0]["source"]}</span>'                                            
                ])
                sentences += '</span>'                
            sentences += '</div>'                
            st.markdown(sentences, unsafe_allow_html=True)
            
            if st.session_state["tts"] == "HuggingFace":
                audio_file = hf_tts(generated_answer)
            else:
                audio_file = google_tts(generated_answer, st.secrets["private_key_id"],
                                        st.secrets["private_key"], st.secrets["client_email"])

            with st.spinner("Generating an audio..."):
                if st.session_state["tts"] == "HuggingFace":
                    with open("out.flac", "wb") as f:
                        f.write(audio_file)
                        st.audio("out.flac")
                else:
                    with open("out.mp3", "wb") as f:
                        f.write(audio_file.audio_content)
                        st.audio("out.mp3")

            st.markdown("""<hr></hr>""", unsafe_allow_html=True)

            model = get_sentence_transformer()

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Context")
            with col2:
                selection = st.selectbox(label='Scope', options=('Paragraphs', 'Sentences', 'Answer Similarity'))

            question_e = model.encode(question, convert_to_tensor=True)
            if selection == "Paragraphs":
                sentences = extract_sentences_from_passages(context_passages)
                context_e = get_sentence_transformer_encoding(sentences)
                scores = util.cos_sim(question_e.repeat(context_e.shape[0], 1), context_e)
                similarity_scores = scores[0].squeeze().tolist()
                for idx, node in enumerate(context_passages):
                    node["answer_similarity"] = "{0:.2f}".format(similarity_scores[idx])
                context_passages = sorted(context_passages, key=lambda x: x["answer_similarity"], reverse=True)
                st.json(context_passages)
            elif selection == "Sentences":
                sentences = extract_sentences_from_passages(context_passages)
                sentences_e = get_sentence_transformer_encoding(sentences)
                scores = util.cos_sim(question_e.repeat(sentences_e.shape[0], 1), sentences_e)
                sentence_similarity_scores = scores[0].squeeze().tolist()
                result = []
                for idx, sentence in enumerate(sentences):
                    result.append(
                        {"text": sentence, "answer_similarity": "{0:.2f}".format(sentence_similarity_scores[idx])})
                context_sentences = json.dumps(sorted(result, key=lambda x: x["answer_similarity"], reverse=True))
                st.json(context_sentences)
            else:
                st.json(sentence_similarity)

        else:
            unknown_error = f"{data}"
            st.warning(unknown_error)