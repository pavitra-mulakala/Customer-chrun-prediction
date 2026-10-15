import os
from dotenv import load_dotenv
from groq import Groq

# Loads GROQ_API_KEY from a .env file sitting next to this script, so you
# don't need to manually set an environment variable in PowerShell every
# time. Create a file named ".env" in this same folder containing:
#   GROQ_API_KEY=gsk_your_actual_key_here
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. Create a '.env' file in this folder containing:\n"
        "  GROQ_API_KEY=gsk_your_actual_key_here\n"
        "Get a key at https://console.groq.com/keys"
    )

client = Groq(api_key=GROQ_API_KEY)

def get_suggestion(provider, dataset_reason, shap_reason):

    prompt = f"""
You are a telecom customer retention expert.

Service Provider:
{provider}

Dataset Reason:
{dataset_reason}

SHAP Reason:
{shap_reason}

Give one short business suggestion (2-3 sentences) for reducing customer churn.
Return only the suggestion.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {e}"