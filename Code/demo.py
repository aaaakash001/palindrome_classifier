import numpy as np
import pandas as pd

import streamlit as st

from palindrome.palindrome_4_fold_oversampling import load_model, predict

# streamlit run demo.py


def get_pred():
    input_str = str(st.session_state.get("input"))
    input_str = input_str.replace(" ", "")
    if input_str == "":
        st.error("Empty string. Please try again with valid input!")
        return

    try:
        x = list(map(int, input_str))
        if len(list(filter(lambda v: v != 0 and v != 1, x))):
            raise ValueError()
    except ValueError:
        st.error(
            f"String contains character other than 0 and 1. Please try again with binary string!"
        )
        return

    if len(x) != 10:
        st.error(
            f'String "{input_str}" is of length {len(x)}. Please enter string of length 10!'
        )
        return

    X = np.array([x]).T
    forward_pass = predict(load_model(), X)
    print(forward_pass)

    is_palindrome = forward_pass[0]
    output_str = ":green[Palindrome]" if is_palindrome else ":red[Not Palindrome]"
    st.write(output_str)


text_input = st.text_input(
    "Input String",
    placeholder="Please enter a binary input string, e.g., 1010000101",
    key="input",
    on_change=get_pred,
)
submit_button = st.button("Submit", type="primary", on_click=get_pred)
