import streamlit as st
from model import SymptomChecker

# Load the model with precautions
checker = SymptomChecker(
    "data/dataset.csv",
    precautions_path="data/symptom_precaution.csv"
)

st.set_page_config(page_title="Symptom Checker", layout="centered")
st.title("🩺 Symptom Checker Chatbot")
st.markdown("<div class='subtitle'>Built an AI-powered assistant that predicts possible conditions and gives health precautions for friends and family— by Urvi Dhomne </div>", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Describe your symptoms (e.g., fever, headache)...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        cleaned_input = user_input.lower().replace(",", " ")
        prediction = checker.predict(cleaned_input)
        precautions = checker.get_precautions(prediction)

        response = f"Based on your symptoms, a possible condition could be: **{prediction}**"

        if precautions:
            response += "\n\n Suggested Precautions:\n"
            for i, p in enumerate(precautions, start=1):
                response += f"- {p}\n"

        response += "\n\n⚠️ This is NOT a diagnosis. Please consult a medical professional."

    except Exception as e:
        response = f"❌ Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
