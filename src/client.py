from text_generation import Client
from config import API_URL, HF_TOKEN

def get_tgi_client():
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    print(headers)
    return Client(API_URL, headers=headers)
