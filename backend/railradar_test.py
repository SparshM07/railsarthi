import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RAILRADAR_API_KEY")


def get_train_data(train_number):
    url = f"https://api.railradar.in/v1/trains/{train_number}/live"

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()["data"]

    return {
        "train": int(data["trainNumber"]),
        "current_station": data["currentLocation"]["stationCode"],
        "next_station": data["nextHalt"]["stationCode"],
        "current_arr_delay": data["currentLocation"]["delayMinutes"]
    }


result = get_train_data(12919)

print(result)