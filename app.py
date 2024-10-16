import streamlit as st
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import json
import os
import PyPDF2

# Streamlit title and description
st.title("Q&A Model with PDF Support")
st.write("Upload a PDF document and enter your query below to ask the Q&A model.")

# Streamlit file uploader for PDFs
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Streamlit input field for the question
question = st.text_input("Enter your question here:")

# Streamlit input fields for AWS credentials
aws_access_key_id = st.text_input("AWS Access Key ID", type="password")
aws_secret_access_key = st.text_input("AWS Secret Access Key", type="password")
region_name = st.text_input("AWS Region", value="us-east-1")

# AWS Bedrock configuration
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

# Function to call Amazon Bedrock model for Q&A
def ask_bedrock(question, context=None):
    try:
        # Define the parameters for the Bedrock request
        model_id = 'meta.llama3-8b-instruct-v1:0'  # Replace with the actual Q&A model ID on Bedrock

        # Prepare input payload for Bedrock model
        payload = {
            "input": f"Context: {context}\n\nQuestion: {question}",
        }

        # Call the Amazon Bedrock model
        response = bedrock_client.invoke_model(
            modelId= "mistral.mistral-large-2402-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )

        # Parse the response
        response_body = json.loads(response['body'].read())
        return response_body.get("output", "No answer found.")

    except (BotoCoreError, ClientError) as error:
        return f"Error invoking Bedrock model: {str(error)}"

# Streamlit button to submit the question
if st.button("Submit"):
    if question:
        if uploaded_file:
            # Extract text from the uploaded PDF
            pdf_text = extract_text_from_pdf(uploaded_file)
            st.write("PDF Content Loaded. Asking the Bedrock model...")
            
            # Get the response from Bedrock with the extracted PDF content as context
            answer = ask_bedrock(question, context=pdf_text)
            st.write(f"Answer: {answer}")
        else:
            st.write("Please upload a PDF file to provide context.")
    else:
        st.write("Please enter a question to ask.")
