import pandas as pd
import streamlit as st

from emotion_model import predict_emotion, train_model

EMOJI = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
}

EXAMPLES = [
    "i feel so happy and grateful today",
    "i feel so lonely and hopeless today",
    "i am so angry at how he treated me",
    "i am terrified of what might happen tomorrow",
]

st.set_page_config(page_title="Emotion Detector", page_icon="🙂")


@st.cache_resource
def get_model():
    return train_model()


model = get_model()

st.title("Emotion Detection using NLP")
st.write(
    "Type a sentence and the model predicts one of six emotions: "
    "joy, sadness, anger, fear, love or surprise."
)

if "text" not in st.session_state:
    st.session_state["text"] = ""

st.write("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state["text"] = example

text = st.text_area("Your text", key="text", height=120)

if st.button("Detect emotion", type="primary"):
    if not text.strip():
        st.warning("Please type some text first.")
    else:
        result = predict_emotion(model, text)
        if result is None:
            st.warning("None of these words are in the model's vocabulary. Try a longer sentence.")
        else:
            top = max(result, key=result.get)
            st.subheader(f"{EMOJI.get(top, '')} {top.capitalize()}")
            st.write(f"Confidence: {result[top]:.0%}")
            chart = pd.Series(result, name="probability").sort_values(ascending=False)
            st.bar_chart(chart)

st.caption("Logistic Regression on bag-of-words (1-2 word phrases). Student project.")
