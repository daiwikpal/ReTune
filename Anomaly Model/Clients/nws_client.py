# nws_client.py
import requests
from typing import Dict, Any, List, Optional
import urllib.parse

BASE_URL = "https://api.weather.gov"

class NWSClient:
    def __init__(self, user_agent: str):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/geo+json"
        })

    def get_point_metadata(self, lat: float, lon: float) -> Dict[str, Any]:
        url = f"{BASE_URL}/points/{lat},{lon}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_hourly_forecast(self, grid_id: str, grid_x: int, grid_y: int) -> Dict[str, Any]:
        url = f"{BASE_URL}/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast/hourly"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_observations(self, station_id: str) -> Dict[str, Any]:
        url = f"{BASE_URL}/stations/{station_id}/observations"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_alerts(self, area: str = "NY") -> Dict[str, Any]:
        url = f"{BASE_URL}/alerts/active?area={area}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def fetch_all_alerts(self, states: List[str], start_time: str, limit: int = 500) -> Dict[str, Any]:
        base_url = f"{BASE_URL}/alerts"
        all_data = []

        area_params = f"area={urllib.parse.quote_plus(','.join(states))}"
        url = f"{base_url}?start={urllib.parse.quote_plus(start_time)}&{area_params}&limit={limit}"

        while url:
            print(f"Fetching: {url}")
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            if "features" in data:
                all_data.extend(data["features"])

            url = data.get("pagination", {}).get("next", None)

        return {
            "features": all_data,
            "total_fetched": len(all_data)
        }
