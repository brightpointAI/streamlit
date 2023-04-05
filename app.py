import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import PyPDF2

st.set_page_config(page_title="PDF to Question App")

# Load the RoBERTa model and tokenizer
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

st.title("PDF to Question App")

# Upload PDF file
uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

if uploaded_file is not None:
    # Convert PDF to text
    pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
    pdf_text = ""
    for page in range(pdf_reader.getNumPages()):
        pdf_text += pdf_reader.getPage(page).extractText()

    # Ask a question
    question = st.text_input("Enter a question about the PDF file:")
    if st.button("Ask"):
        # Tokenize the input
        inputs = tokenizer(question, pdf_text, add_special_tokens=True, return_tensors="pt")

        # Get the model's predictions
        answer_start_scores, answer_end_scores = model(**inputs).values()

        # Get the most likely beginning and ending of the answer
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        # Get the answer text
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

        st.write("Answer:", answer)
