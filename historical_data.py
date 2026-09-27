import pandas as pd
from io import BytesIO
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"

LEAGUES = {
    "Primera División": "SP1",
    "Segunda División": "SP2",
}

def load_season(league, season):
    code = LEAGUES[league]
    url = f"{BASE_URL}/{season}/{code}.csv"

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    return pd.read_csv(BytesIO(response.content))


def load_history(league, seasons):
    frames = []

    for season in seasons:
        try:
            df = load_season(league, season)
            df["Season"] = season
            frames.append(df)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)