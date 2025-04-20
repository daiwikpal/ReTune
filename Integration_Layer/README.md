# Model Integration Layer

This application provides an integration layer between different weather prediction models, combining their outputs using Bayesian model averaging to produce a final precipitation prediction.

## Overview

The integration layer currently supports:
- Fetching predictions from the anomaly model
- Preparing for integration with the precipitation model
- Combining predictions using weighted Bayesian model averaging

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from model_integration import ModelIntegrator

# Initialize the integrator with the URL of the anomaly model
integrator = ModelIntegrator(anomaly_model_url="http://localhost:8000")

# Get a prediction for a specific month
result = integrator.get_integrated_prediction(target_month="2023-05")
print(result)
```

### With Precipitation Model Results

If you have a prediction from the precipitation model, you can incorporate it:

```python
from model_integration import ModelIntegrator

integrator = ModelIntegrator(anomaly_model_url="http://localhost:8000")

# Provide both the target month and a precipitation model prediction
result = integrator.get_integrated_prediction(
    target_month="2023-05",
    precipitation_prediction=2.5,  # Example value
    anomaly_weight=0.6  # Giving 60% weight to anomaly model, 40% to precipitation model
)
print(result)
```

## API Reference

### ModelIntegrator

The main class that handles the integration of different models.

#### Methods:

- `__init__(anomaly_model_url)`: Initializes the integrator with the URL of the anomaly model.
- `get_anomaly_prediction(target_month)`: Gets a prediction from the anomaly model for the specified target month.
- `bayesian_model_averaging(anomaly_prediction, precipitation_prediction, anomaly_weight)`: Combines predictions using Bayesian model averaging.
- `get_integrated_prediction(target_month, precipitation_prediction, anomaly_weight)`: Gets an integrated prediction combining both models.

## Example

The application includes a simple example that demonstrates how to use the integration layer:

```bash
python model_integration.py
```

This will show predictions with and without the precipitation model input.

## Logging

The application logs all activities to both the console and a file named `integration_logs.log`. You can monitor this file for debugging and tracking purposes. 