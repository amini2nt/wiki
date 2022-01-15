import streamlit as st

settings = {}


def app():
    with st.form("settings"):
        st.write("LFQA model parameters")

        settings["min_length"] = st.slider("Min length", 20, 80, st.session_state["min_length"],
                                           help="Min response length (words)")
        settings["max_length"] = st.slider("Max length", 128, 320, st.session_state["max_length"],
                                           help="Max response length (words)")
        settings["do_sample"] = st.checkbox("Use sampling", st.session_state["do_sample"],
                                            help="Whether or not to use sampling ; use greedy decoding otherwise.")
        settings["early_stopping"] = st.checkbox("Early stopping", st.session_state["early_stopping"],
                                                 help="Whether to stop the beam search when at least num_beams sentences are finished per batch or not.")
        settings["num_beams"] = st.slider("Num beams", 1, 16, st.session_state["num_beams"],
                                          help="Number of beams for beam search. 1 means no beam search.")
        settings["temperature"] = st.slider("Num beams", 0.0, 1.0, st.session_state["temperature"], step=0.1,
                                            help="The value used to module the next token probabilities")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            for k, v in settings.items():
                st.session_state[k] = v

