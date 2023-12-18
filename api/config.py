from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())
class Config:
    SECRET_KEY=os.getenv('SECRET_KEY')