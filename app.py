import streamlit as st
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import json
import os
# from dotenv import load_dotenv

# Load environment variables (for AWS credentials)
# load_dotenv()

# Streamlit title and description
st.title("Q&A Model")
st.write("Enter your query below to ask the Q&A model.")

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
            aws_secret_access_key=aws_secret_access_key )
    
# Function to call Amazon Bedrock model for Q&A
def ask_bedrock(question):
    try:
        # Define the parameters for the Bedrock request
        model_id = 'amazon.titan-embed-text-v1'  # Replace this with the actual Q&A model ID on Bedrock

        # Prepare input payload for Bedrock model
        payload = {
            "input": question,
        }

        # Call the Amazon Bedrock model
        response = bedrock_client.invoke_model(
            modelId=model_id,
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
        st.write("Asking the Bedrock model...")
        # Get the response from Bedrock
        answer = ask_bedrock(question)
        # Display the answer
        st.write(f"Answer: {answer}")
    else:
        st.write("Please enter a question to ask.")


# Streamlit execution command
# To run the app, use: streamlit run <filename>.py
