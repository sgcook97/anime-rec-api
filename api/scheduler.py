import schedule
import time
import requests

def update_model():
    requests.post("http://some-url/api/update-model")

schedule.every().monday.at("02:00").do(update_model)

while True:
    schedule.run_pending()
    time.sleep(60)