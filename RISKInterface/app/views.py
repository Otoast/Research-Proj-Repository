from .weather_functions import *
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import requests, json
# Create your views here.



def home(request): return render(request, "home_page.html")
def basic_RISK_calculator(request): return render(request, "basic_risk_calculator.html")
def location_RISK_calculator(request): return render(request, "location_risk_calculator.html")

def parse_RISK_submit(request):
    request_body = json.loads(request.body)
    rq_keys = request_body.keys()
    if 'type' not in rq_keys or request_body['type'] not in ['general', 'location']:
        json_response = JsonResponse({'message': 'Improper RISK Calculation Type'}, status=400)
    request_type = request_body['type']
    if request_type == 'general':
        json_response = _handle_general(request)
    elif request_type == 'location':
        json_response = _handle_location(request)
        
    return json_response

def _handle_general(request):
    degrees = request.POST.get('degreeInput')
    windspeed = request.POST.get('windSpeedInput')
    humidity = request.POST.get('humidityInput')
    if not (valid_input([degrees, windspeed, humidity])):
        return JsonResponse({'message': 'Bad Input Data Recieved'}, status=400)
    
    degrees, windspeed, humidity = map(float, (degrees, windspeed, humidity))

    response_data = get_response(degrees, humidity, windspeed)
    return JsonResponse(response_data)

def _handle_location(request):
    if request.method != 'POST':
        return JsonResponse({'message': 'Non POST request sent for calculation'}, status=400)
    body = json.loads(request.body)
    latitude = body['lat']
    longitude = body['lng']
    zip_code = body['zip']
    lat_lon_empty = latitude == 'NA' and longitude == 'NA' 
    zip_empty = zip_code == 'NA'
    
    location = None
    if  lat_lon_empty and not zip_empty:
        location = f'{zip_code}'
    elif not lat_lon_empty and zip_empty:
        location=f'{latitude},{longitude}'

    url = f"https://api.weatherapi.com/v1/current.json?key={WEATHER_FUNC_API_KEY}&q={location}"
    response = requests.get(url)
    if response.status_code != 200:
        return JsonResponse({'message': 'Bad Data entered'}, status=400)

    weather_data = response.json()
    
    temperature = weather_data['current']['temp_f']
    humidity = weather_data['current']['humidity']
    wind_speed = weather_data['current']['wind_mph']
    response_data = get_response(temperature, humidity, wind_speed)

    l = weather_data['location']
    location = f'{l['name']}, {l['region']}, {l['country']}'

    response_data['location'] = location
    response_data['lat'] = l['lat']
    response_data['lng'] = l['lon']

    return JsonResponse(response_data)

def get_response(temperature, humidity, wind_speed):
    d_category, d_effect = get_temp_rating(temperature)
    h_category, h_effect = get_humidity_rating(humidity)
    w_category, w_effect = get_wind_rating(wind_speed)  
    response_data = dict()
    response_data['RESPONSE_degrees'] = {
        'value' : temperature,
        'category' : d_category,
        'description' : d_effect
    }
    response_data['RESPONSE_humidity'] = {
        'value' : humidity,
        'category' : h_category,
        'description' : h_effect
    }
    response_data['RESPONSE_windspeed'] = {
        'value' : wind_speed,
        'category' : w_category,
        'description' : w_effect
    }
    response_data['risk'] = get_risk_rating(temperature, humidity, wind_speed)
    response_data['risk'] = get_risk_rating(temperature, humidity, wind_speed)
    response_data['content'] = get_weather_condition_html(response_data)
    response_data['safety_recc'] = get_safety_reccomendation(response_data['risk'])
    return response_data