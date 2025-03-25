import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

def call_claude(query, context, image_context=None):
    """
    For sending a request to AWS Bedrock to generate a response from Claude.
    """
        # Combine the context documents into a single text block to include in the prompt.
    context_text = "\n\n".join([doc.page_content for doc in context])
    
    if image_context:
        context_text = image_context + "\n\n" + context_text

    # Create a structured input for the model with a clear prompt, context, and user query.
    input_text = (
        system_prompt +
        "\nContext: " + context_text + 
        "\n\nUser: " + query + 
        "\nBot:"
    )

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2500,
        "messages": [
            {
                "role": "user",
                "content": input_text
            }
        ]
    }

    response = bedrock_client.invoke_model(
        body=json.dumps(payload),
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        contentType="application/json",
        accept="application/json"
    )
    
    response_body = json.loads(response['body'].read())
    answer = response_body['content'][0]['text']

    return answer.strip().split("\nBot:")[-1].strip()

def call_titan(prompt: str):
    """
    For sending a request to AWS Bedrock to generate a response from Titan Express.
    """
    payload = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 200,
            "stopSequences": [],
            "temperature": 0.3,
            "topP": 0.4
        }
    }

    response = bedrock_client.invoke_model(
        body=json.dumps(payload),
        modelId="amazon.titan-text-express-v1",
        contentType="application/json",
        accept="application/json"
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['results'][0]['outputText']

if __name__ == "__main__":
    print(call_titan("Say hi in one word"))