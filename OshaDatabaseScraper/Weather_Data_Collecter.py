import requests
import re
import pandas as pd
from time import sleep
from bs4 import BeautifulSoup
from random import randrange
from numpy import NaN
from os import getenv
import datetime
from tqdm import tqdm


def _best_fit(zip, zip_codes):
    best_zip_option = None
    abs_difference = 9999
    options = zip_codes[zip_codes['zip'].str[0:len(zip) - 1] == zip[:-1]]
    if options.empty:   raise RuntimeError(f"Potentially invalid zip code: {zip}.")
    for zip_option in options['zip']:
        abs_diff = abs(int(zip_option[-2:]) - int(zip[-2:])) 
        if abs_diff < abs_difference: # If the difference in zip code difference is less than previous thought best
            best_zip_option = zip_option
            abs_difference = abs_diff
    return zip_codes[zip_codes['zip'] == best_zip_option]

def _US_location(option):   
    return  any(["US" in x['short_name'] for x in option['address_components']])   

def _ask_google_maps(zip_code:str, site_address:str=None) -> dict:
    """
        If you aren't confident that the zip code you're inserting is a US zip code, please also include a site address for the function to default to.
    """
    zip_code = str(zip_code).replace('\n', ', ')
    api_call = 'https://maps.googleapis.com/maps/api/geocode/json?parameters'
    
    GOOGLE_API_KEY = getenv('GOOGLE_API_KEY')
    SECRET = getenv('GOOGLE_API_SECRET')

    params = {
        'address': zip_code + "zip code",
        'key': GOOGLE_API_KEY,
    }
    response = requests.get(api_call, params=params)
    response.raise_for_status()
    response = response.json()

    any_option_US_location = any([_US_location(x) for x in response['results']])
    if response['status'] == 'ZERO_RESULTS' or not any_option_US_location:
        params['address'] = site_address
        response = requests.get(api_call, params=params)
        response.raise_for_status()
        response = response.json()
    if response['status'] != 'OK': raise RuntimeError(f"Error calling Google Geocoding API. Status was {response['status']}.\nZip Code was: {zip_code}\nAddress was:\n{site_address}")
    
    final_choice = None
    for option in response['results']: # G maps sometimes returns multiple plausible location in response
        if _US_location(option):
            final_choice = option
            break
        
    if final_choice is None:    
        print(f"Warning: Potential Ambiguous or Non-US Location. Response Was :\n{response}")
        final_choice = response['results'][0]
    return final_choice['geometry']['location']

def get_longitude_latitude(zip_codes:pd.DataFrame, zip_code:str, site_address:str):
    try:
        result = zip_codes[zip_codes['zip'] == str(zip_code)]
        if result.empty:
            result = _best_fit(zip_code, zip_codes)           
        result = {'lat' : result['lat'].values[0], 'lng' : result['lng'].values[0]}
    except RuntimeError:
        result = _ask_google_maps(zip_code=zip_code, site_address=site_address)
    return result

def get_open_meteo_data(longitude, latitude, start_date, end_date):
    weather_link = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&"
        "hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,"
        "precipitation,rain,snowfall,snow_depth,weather_code,pressure_msl,surface_pressure,"
        "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
        "et0_fao_evapotranspiration,vapour_pressure_deficit,wind_speed_10m,wind_speed_100m,"
        "wind_direction_10m,wind_direction_100m,wind_gusts_10m&timeformat=unixtime&timezone=auto"
    )
    response = requests.get(weather_link)
    response.raise_for_status()
    response:dict = response.json()['hourly']
    return pd.DataFrame(response)

def parse_epoch(date:str, time:str):
    is_afternoon = "p.m." in time
    time = time.replace("a.m.", '').replace('p.m.', '')
    hour, minute = map(int, time.split(sep=':'))
    
    if is_afternoon and hour != 12:         hour += 12
    elif not is_afternoon and hour == 12:   hour = 0
    if minute >= 30:                        hour += 1

    date = datetime.datetime.strptime(date, "%m/%d/%Y") + datetime.timedelta(hours=hour)

    return int(date.timestamp())



def main():
    accident_table = pd.read_csv("accident_data.csv", dtype=str)
    accident_table = accident_table[~accident_table['Time of Incident'].isna()]
    accident_table.reset_index(inplace=True, drop=True)
    zip_codes = pd.read_csv("simplemaps_uszips_basicv1.85/uszips.csv", converters={'zip':str})

    print("Obtaining Weather Data...")
    weather_table = list()
    accident_table = pd.read_csv("accident_data.csv", dtype=str)
    accident_table = accident_table[~accident_table['Time of Incident'].isna()]
    accident_table.reset_index(inplace=True, drop=True)
    zip_codes = pd.read_csv("simplemaps_uszips_basicv1.85/uszips.csv", converters={'zip':str})

    weather_table = list()
    for i in tqdm(range(accident_table.shape[0])):
        summary_nr = accident_table['Summary Nr'][i]
        time_of_incident = accident_table['Time of Incident'][i]
        zip_code = accident_table['Zip Code'][i]
        site_address = accident_table['Site Address'][i]
        start_date = accident_table['Event Date'][i]

        start_date = datetime.datetime.strptime(start_date, "%m/%d/%Y")
        end_date =  (start_date + datetime.timedelta(1)).strftime("%Y-%m-%d")
        start_date = (start_date - datetime.timedelta(1)).strftime("%Y-%m-%d")

        epoch = parse_epoch(accident_table['Event Date'][i], accident_table['Time of Incident'][i])
        location = get_longitude_latitude(zip_codes=zip_codes, zip_code=zip_code, site_address=site_address)
        weather_data = get_open_meteo_data(longitude=location['lng'], latitude=location['lat'], start_date=start_date, end_date=end_date)
        weather_data = weather_data[weather_data['time'] == epoch]
        if weather_data.empty:
            print(start_date)
            print(end_date)
            print(epoch)
            print(weather_data)
            raise RuntimeError("BAD SEARCH")
        weather_data.fillna(value=0, inplace=True)
        weather_data.insert(0, 'Summary Nr', [summary_nr])
        weather_table.append(weather_data)

    weather_table = pd.concat(weather_table, axis=0, ignore_index=True)
    print(weather_table)


    complete_table = pd.merge(accident_table, weather_table, on=['Summary Nr'])
    complete_table.to_csv("Accident_Weather_Data_Full.csv", index=False, encoding='utf-8')


if __name__ == '__main__':
    main()