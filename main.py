import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")


def generate_simulated_data(start_date, weeks, params):
    """
    Generates simulated weekly data for temperature or visitor traffic.

    Args:
        start_date (str): The starting date for the data generation ('YYYY-MM-DD').
        weeks (int): The total number of weeks to generate.
        params (dict): A dictionary containing parameters for the simulation.
                       Required keys: 'type', 'base', 'peak', 'noise'.

    Returns:
        pandas.Series: A Series with a DatetimeIndex and simulated data.
    """
    data = []
    dates = pd.date_range(start=start_date, periods=weeks, freq='W')

    for i in range(weeks):
        week_of_year = dates[i].isocalendar()[1]
        value = 0

        # --- Logic for different data types ---
        if params['type'] == 'temperature':
            # Smooth sine wave for temperature seasonality
            amplitude = (params['peak'] - params['base']) / 2
            midpoint = params['base'] + amplitude
            # Shift sine wave so winter (mid-year) is the trough
            value = midpoint - amplitude * np.cos(2 * np.pi * (week_of_year - 2) / 52)
            value += (np.random.random() - 0.5) * params['noise']

        elif params['type'] == 'traffic':
            # Sharp peak for visitor traffic in winter
            peak_start = 23  # Early June
            peak_end = 38  # Mid-September
            peak_duration = peak_end - peak_start

            if peak_start <= week_of_year <= peak_end:
                position_in_peak = (week_of_year - peak_start) / peak_duration
                peak_factor = np.sin(position_in_peak * np.pi)
                value = params['base'] + peak_factor * (params['peak'] - params['base'])
            else:
                value = params['base']

            value += (np.random.random() - 0.5) * params['noise']

        data.append(max(0, value))  # Ensure no negative values

    return pd.Series(data, index=dates)


def create_and_plot_forecast(config):
    """
    Runs the full process: data generation, model fitting, forecasting, and plotting.

    Args:
        config (dict): A configuration dictionary for a specific forecast.
                       Required keys: 'location', 'data_type', 'unit', and a 'params' dict.
    """
    print(f"--- Running forecast for {config['location']} {config['data_type']} ---")

    # 1. Generate historical data (2 years)
    hist_data = generate_simulated_data('2024-01-01', 104, config['params'])

    # 2. Fit the SARIMA model
    # The model (p,d,q)(P,D,Q,s) is chosen to handle seasonality.
    model = SARIMAX(hist_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
    fit_result = model.fit(disp=False)

    # 3. Forecast for 2026 (52 weeks)
    forecast = fit_result.get_forecast(steps=52)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # 4. Plotting the results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(hist_data.index, hist_data, label='Historical Data', color='gray')
    ax.plot(forecast_mean.index, forecast_mean, label='Forecast (2026)', color='blue', linestyle='--')
    ax.fill_between(forecast_ci.index,
                    forecast_ci.iloc[:, 0],
                    forecast_ci.iloc[:, 1], color='blue', alpha=0.1, label='95% Confidence Interval')

    ax.set_title(f"Simulated ARIMA Forecast: {config['location']} Weekly {config['data_type']} for 2026", fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f"Weekly {config['data_type']} ({config['unit']})", fontsize=12)
    ax.legend(fontsize=10)

    # Save the figure
    filename = f"{config['location'].replace(' ', '_')}_{config['data_type']}_forecast_2026.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close(fig)  # Close the figure to free up memory


if __name__ == '__main__':
    # --- Configuration List ---
    # Define all the forecasts you want to run here.
    # It's easy to add, remove, or modify forecasts without changing the functions.

    forecast_configs = [
        {
            'location': 'Mount Buller',
            'data_type': 'Temperature',
            'unit': '°C',
            'params': {'type': 'temperature', 'base': -2, 'peak': 15, 'noise': 4}
        },
        {
            'location': 'Perisher',
            'data_type': 'Temperature',
            'unit': '°C',
            'params': {'type': 'temperature', 'base': -5, 'peak': 12, 'noise': 5}
        },
        {
            'location': 'Hotham',
            'data_type': 'Temperature',
            'unit': '°C',
            'params': {'type': 'temperature', 'base': -6, 'peak': 11, 'noise': 5}
        },
        {
            'location': 'Hotham',
            'data_type': 'Visitor Traffic',
            'unit': 'Visitors',
            'params': {'type': 'traffic', 'base': 500, 'peak': 15000, 'noise': 2000}
        },
        {
            'location': 'Mount Buller',
            'data_type': 'Visitor Traffic',
            'unit': 'Visitors',
            'params': {'type': 'traffic', 'base': 600, 'peak': 18000, 'noise': 2500}
        },
        {
            'location': 'Perisher',
            'data_type': 'Visitor Traffic',
            'unit': 'Visitors',
            'params': {'type': 'traffic', 'base': 800, 'peak': 22000, 'noise': 3000}
        }
    ]

    # Loop through the configurations and run each forecast
    for config in forecast_configs:
        create_and_plot_forecast(config)

    print("\n--- All forecasts completed. ---")
