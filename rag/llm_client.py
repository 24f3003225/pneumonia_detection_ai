from config import GEMINI_API_KEY, GEMINI_MODEL

def llm_generate(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY is not set."

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return (resp.text or "").strip()