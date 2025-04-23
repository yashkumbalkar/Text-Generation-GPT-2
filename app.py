import streamlit as st
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set device
device = "cuda"


st.set_page_config(page_title="GPT-2 Text Generator", layout="centered")
st.title("GPT-2 Text Generator")
st.markdown("Type a prompt and let GPT-2 do the rest.")


# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model

tokenizer, model = load_model()


# Text Input
input_text = st.text_area("Enter your prompt here: ", height=150, placeholder="Type something here...")

# Generate Button
if st.button("Generate Text"):
  if input_text.strip():
    # Tokenize input and get generated text
    max_length = 128
    input_ids = tokenizer(input_text, return_tensors="pt")
    input_ids = input_ids['input_ids'].to(device)

    # Text Generation
    output = model.generate(input_ids, max_length=max_length, do_sample=True, top_p=1)
    original_text = tokenizer.decode(output[0])
    generated_text = original_text.replace('\n', ' ').strip()

    if '<|endoftext|>' in generated_text:
      clean_text = generated_text.replace('<|endoftext|>','')

    # Display the prediction result
    st.markdown("### 💬 Generated Output:")
    st.success(clean_text)
  else:
    st.error("Please enter some text")

# Footer
st.markdown("""
    Feel free to share and use this app!
""")

