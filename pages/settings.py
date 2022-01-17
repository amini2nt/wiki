import streamlit as st

settings = {}


def app():
    st.markdown("""
        <style>
            div[data-testid="stForm"] {
                border: 0;
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
            footer {
                display: none !important;
            }
            .footer-custom a {
                color: #70707066;
            }
            h1 {
                margin-bottom: 50px
            }
            button[kind="formSubmit"]{
                margin-top: 50px;
                border-radius: 20px;
                padding: 5px 20px;
                font-size: 18px;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.form("settings"):
        footer = """
            <div class="footer-custom">
                Streamlit app created by <a href="https://www.linkedin.com/in/danijel-petkovic-573309144/" target="_blank">Danijel Petkovic</a>
            </div>
        """
        st.markdown(footer, unsafe_allow_html=True)

        st.title("LFQA model parameters")

        settings["min_length"] = st.slider("Min length", 20, 80, st.session_state["min_length"],
                                           help="Min response length (words)")
        st.markdown("""<hr></hr>""", unsafe_allow_html=True)
        settings["max_length"] = st.slider("Max length", 128, 320, st.session_state["max_length"],
                                           help="Max response length (words)")
        st.markdown("""<hr></hr>""", unsafe_allow_html=True)
        settings["do_sample"] = st.checkbox("Use sampling", st.session_state["do_sample"],
                                            help="Whether or not to use sampling ; use greedy decoding otherwise.")
        st.markdown("""<hr></hr>""", unsafe_allow_html=True)
        settings["early_stopping"] = st.checkbox("Early stopping", st.session_state["early_stopping"],
                                                 help="Whether to stop the beam search when at least num_beams sentences are finished per batch or not.")
        st.markdown("""<hr></hr>""", unsafe_allow_html=True)
        settings["num_beams"] = st.slider("Num beams", 1, 16, st.session_state["num_beams"],
                                          help="Number of beams for beam search. 1 means no beam search.")
        st.markdown("""<hr></hr>""", unsafe_allow_html=True)
        settings["temperature"] = st.slider("Num beams", 0.0, 1.0, st.session_state["temperature"], step=0.1,
                                            help="The value used to module the next token probabilities")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            for k, v in settings.items():
                st.session_state[k] = v
