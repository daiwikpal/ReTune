import requests
import json
import urllib.parse

def fetch_all_alerts(states, start_time, limit=500):
    base_url = "https://api.weather.gov/alerts"
    all_data = []

    # URL encode the 'area' parameter as a comma-separated list (e.g., area=NY,NJ,PA)
    area_params = f"area={urllib.parse.quote_plus(','.join(states))}"

    # Start with initial URL
    url = f"{base_url}?start={urllib.parse.quote_plus(start_time)}&{area_params}&limit={limit}"
    headers = {"accept": "application/geo+json"}

    while url:
        print(f"Fetching: {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if "features" in data:
            all_data.extend(data["features"])

        # Follow pagination if exists
        pagination = data.get("pagination", {})
        url = pagination.get("next", None)

    return {
        "features": all_data,
        "total_fetched": len(all_data)
    }

# Example usage:
if __name__ == "__main__":
    states = ["NY", "NJ"]
    start_time = "2015-01-01T00:00:00Z"
    result = fetch_all_alerts(states, start_time)
    print(f"Fetched {result['total_fetched']} alerts.")

    # Write results to a file
    with open("alerts_results.json", "w") as file:
        json.dump(result, file, indent=4)
    print("Results written to alerts_results.json")
