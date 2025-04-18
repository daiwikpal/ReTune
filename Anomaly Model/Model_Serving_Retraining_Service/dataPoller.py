# data_poller.py
import csv
import sys
from datetime import datetime, timedelta
from Clients.nws_client import NWSClient

LAT_NYC = 40.7128
LON_NYC = -74.0060
STATION_ID = "KNYC"
USER_AGENT = "myweatherapp.com, contact@myweatherapp.com"


def poll_data(start_time: str, end_time: str):
    client = NWSClient(user_agent=USER_AGENT)

    # Get grid info
    point_metadata = client.get_point_metadata(LAT_NYC, LON_NYC)
    grid_id = point_metadata['properties']['gridId']
    grid_x = point_metadata['properties']['gridX']
    grid_y = point_metadata['properties']['gridY']

    # Get hourly forecast (this will only include recent/present forecast)
    forecast = client.get_hourly_forecast(grid_id, grid_x, grid_y)
    forecast_periods = forecast['properties']['periods']

    # Get observations in time range
    obs_url = f"https://api.weather.gov/stations/{STATION_ID}/observations?start={start_time}&end={end_time}"
    observations = client.session.get(obs_url)
    observations.raise_for_status()
    obs_data = observations.json().get('features', [])

    # Get alerts (alerts only available for past 7 days, historical must use archive)
    alerts_url = f"https://api.weather.gov/alerts?area=NY&start={start_time}&end={end_time}"
    alerts = client.session.get(alerts_url)
    alerts.raise_for_status()
    alerts_data = alerts.json().get('features', [])

    # Write forecast to CSV
    with open("nyc_forecast.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["startTime", "temperature", "windSpeed", "shortForecast"])
        for period in forecast_periods:
            writer.writerow([period['startTime'], period['temperature'], period['windSpeed'], period['shortForecast']])

    # Write observations to CSV
    with open("nyc_observations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "temperature", "windSpeed", "humidity"])
        for obs in obs_data:
            props = obs['properties']
            writer.writerow([
                props.get('timestamp'),
                props.get('temperature', {}).get('value'),
                props.get('windSpeed', {}).get('value'),
                props.get('relativeHumidity', {}).get('value')
            ])

    # Write alerts to CSV
    with open("nyc_alerts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event", "severity", "headline", "description"])
        for alert in alerts_data:
            props = alert['properties']
            writer.writerow([
                props.get('event'),
                props.get('severity'),
                props.get('headline'),
                props.get('description')
            ])

    states = ["NY", "NJ", "PA", "CT"]
    start_time = "2015-01-01T00:00:00Z"
    result = client.fetch_all_alerts(states, start_time)

    print(f"Fetched {result['total_fetched']} alerts.")
    with open("alerts_results.json", "w") as file:
        import json
        json.dump(result, file, indent=4)
    print("Results written to alerts_results.json")

def main():
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=5*365)

    start_time = start_date.isoformat() + "Z"
    end_time = end_date.isoformat() + "Z"

    print(f"Fetching data from {start_time} to {end_time}...")
    poll_data(start_time, end_time)

    print("Data fetched successfully.")

if __name__ == "__main__":
    main()
