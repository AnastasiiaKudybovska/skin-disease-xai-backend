import os
from dotenv import load_dotenv

load_dotenv()

origins = [
    "http://localhost:3000",
]

SECRET_KEY: str = os.getenv("SECRET_KEY", "secret_key")