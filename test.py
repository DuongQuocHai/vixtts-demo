import os
import sys
import requests
import re


def convert_to_lat_lng(input_string, google_map_api_key, country_code='vn'):
    """
    Convert a string (text address or lat,lng) to latitude and longitude coordinates.
    
    Args:
        input_string (str): Either a text address or a lat,lng pair (e.g., "40.7128,-74.0060").
        google_map_api_key (str): Google Maps API key.
        country_code (str): ISO country code (default: 'vn').
    
    Returns:
        dict: Contains 'lat' and 'lng' if successful, or 'error' if failed.
    """
    def get_language_code():
        return 'en'

    lat_lng_pattern = r'^-?\d+\.\d+,-?\d+\.\d+$'
    is_lat_lng = re.match(lat_lng_pattern, input_string.strip())

    if is_lat_lng:
        try:
            lat, lng = map(float, input_string.split(','))
            url = (
                f'https://maps.googleapis.com/maps/api/geocode/json?'
                f'latlng={lat},{lng}&key={google_map_api_key}&language={get_language_code()}'
            )
            response = requests.get(url)
            json_data = response.json()

            if json_data.get('status') == 'OK':
                location = json_data['results'][0]['geometry']['location']
                return {'lat': location['lat'], 'lng': location['lng']}
            else:
                return {'error': json_data}
        except Exception as e:
            return {'error': str(e)}
    else:
        try:
            encoded_input = requests.utils.quote(input_string.strip())
            url = (
                f'https://maps.googleapis.com/maps/api/place/autocomplete/json?'
                f'input={encoded_input}&key={google_map_api_key}&language={get_language_code()}'
                f'&components=country:{country_code.lower()}'
            )
            response = requests.get(url)
            json_data = response.json()

            if json_data.get('status') == 'OK' and json_data.get('predictions'):
                place_id = json_data['predictions'][0]['place_id']
                geocode_url = (
                    f'https://maps.googleapis.com/maps/api/geocode/json?'
                    f'place_id={place_id}&key={google_map_api_key}'
                )
                geo_response = requests.get(geocode_url)
                geo_json = geo_response.json()

                if geo_json.get('status') == 'OK':
                    location = geo_json['results'][0]['geometry']['location']
                    return {'lat': location['lat'], 'lng': location['lng']}
                else:
                    return {'error': geo_json}
            else:
                return {'error': json_data}
        except Exception as e:
            return {'error': str(e)}


if __name__ == '__main__':
    api_key = os.getenv('GOOGLE_MAP_API_KEY')
    if not api_key:
        print("Please set the GOOGLE_MAP_API_KEY environment variable.")
        sys.exit(1)

    test_cases = [
        "40.7128,-74.0060",
        "Hanoi, Vietnam",
        "1600 Amphitheatre Parkway, Mountain View, CA"
    ]

    for case in test_cases:
        result = convert_to_lat_lng(case, api_key)
        print(f"Input: {case}")
        print(f"Result: {result}")
        print('-' * 40) 