# Add api key from openweather api
WEATHER_FUNC_API_KEY = None


HUMIDITY_CATEGORIES = {
    'Very Dry' : 'Can cause dryness and irritation of the skin, throat, and eyes. May also increase static electricity and respiratory discomfort.',
    'Comfortable' : 'Ideal indoor relative humidity range. Most people feel comfortable within this range, and it helps to reduce the growth of mold and dust mites.',
    'Slightly Humid' :'Can start to feel a bit uncomfortable, especially in higher temperatures. May promote mold growth and dust mites.',
    'Humid': 'Generally uncomfortable, particularly at higher temperatures. Increased risk of mold growth and dust mites. Can also make it harder to cool down.',
    'Very Humid' : 'Feels oppressive and very uncomfortable. High risk of mold growth and dust mites. Can significantly impair the body\'s ability to cool down.',
    'Extremely Humid' : 'Extremely uncomfortable and oppressive. High likelihood of heat stress and heat-related illnesses. Very high risk of mold growth and dust mites.'
}

TEMPERATURE_CATEGORIES = {
    'Extreme Cold': 'High risk of hypothermia and frostbite. Protective clothing is essential. Wind chill can make it feel much colder than the actual temperature.',
    'Cold': 'Risk of hypothermia if exposed for long periods without proper clothing. Generally uncomfortable for extended outdoor activities without protection.',
    'Cool': 'Mildly uncomfortable without appropriate clothing. Safe for most outdoor activities with proper attire.',
    'Mild': 'Comfortable for most outdoor activities. Light to moderate clothing is suitable.',
    'Warm': 'Generally comfortable. Ideal for outdoor activities.',
    'Hot': 'Can be uncomfortable, especially with high humidity. Heat-related illnesses can occur with prolonged exposure and activity.',
    'Very Hot': 'High risk of heat-related illnesses. Limit strenuous activities and ensure adequate hydration.',
    'Extreme Heat': 'Severe risk of heat-related illnesses. Avoid strenuous activities, seek shade, and stay hydrated.'
}

WIND_CATEGORIES = {
    0: 'Calm',
    1: 'Light Air',
    2: 'Light Breeze',
    3: 'Gentle Breeze',
    4: 'Moderate Breeze',
    5: 'Fresh Breeze',
    6: 'Strong Breeze',
    7: 'Near Gale',
    8: 'Gale',
    9: 'Strong Gale',
    10: 'Storm',
    11: 'Violent Storm',
    12: 'Hurricane Force'
}



def get_weather_condition_html(d):
    return f"""
        <div id="temp-response">
            <p id="c1">
                <strong>Temperature Rating: </strong>{d['RESPONSE_degrees']['category']}</p>
            <p id="d1">
                <strong>Description: </strong>{d['RESPONSE_degrees']['description']}</p>
        </div>
        <br/>
        <div id="humidity-response">
            <p id="c1">
                <strong>Humidity Rating: </strong>{d['RESPONSE_humidity']['category']}</p>
            <p id="d1">
                <strong>Description: </strong>{d['RESPONSE_humidity']['description']}</p>
        </div>
        <br/>
        <div id="windspeed-response">
            <p id="c1">
                <strong>Wind Speed Rating: </strong>{d['RESPONSE_windspeed']['category']}</p>
            <p id="d1">
                <strong>Description: </strong>{d['RESPONSE_windspeed']['description']}</p>
        </div>
        <br/>
    """

def get_safety_reccomendation(risk):
    return "Coming soon!"





def valid_input(numerical_values):
    for v in numerical_values:
        v = str(v)
        if v is None: return False
        new_v = v.replace('-', "").replace('.', "")
        if not new_v.isdecimal() or len(v) > 4:
            return False
        c = v.count('-')
        if (c> 1 or (c == 1 and v[0] != '-') ) or ( v.count('.') > 1 ):
            return False
    return True
        
def get_humidity_rating(humidity):
    key = None
    options = ['Very Dry', 'Comfortable', 'Slightly Humid', 'Humid', 'Very Humid', 'Extremely Humid']
    for i, extrema in enumerate([30, 50, 60, 70, 80]):
        if humidity <= extrema:
            key = options[i]
            return key, HUMIDITY_CATEGORIES[key]
    
    key = options[-1]
    return key, HUMIDITY_CATEGORIES[key]

def get_temp_rating(degrees):
        key = None
        options = ['Extreme Cold', 'Cold', 'Cool', 'Mild', 'Warm', 'Hot', 'Very Hot', 'Extreme Heat']
        for i, extrema in enumerate([32, 50, 59, 70, 81, 90, 100]):
            if degrees <= extrema:
                key = options[i]
                return key, TEMPERATURE_CATEGORIES[key]
        key = options[-1]
        return key, TEMPERATURE_CATEGORIES[key]
def get_wind_rating(windspeed):
    beaufort_number = 0 
    for extrema in [1, 3, 7, 12, 18, 24, 131, 38, 46, 54, 63, 72]:
        if windspeed > extrema:
            beaufort_number += 1
        else:
            break
    return beaufort_number, WIND_CATEGORIES[beaufort_number]

def get_risk_rating(temperature, humidity, windspeed):
    if (temperature > 90 or
        humidity > 70 or
        windspeed > 20):
        return'High'

    # Moderate Risk Condition
    elif (80 <= temperature <= 90 or 50 <= humidity <= 70 or 10 <= windspeed <= 20):
        return 'Moderate'

    # Low Risk Condition
    else:
       return 'Low'
