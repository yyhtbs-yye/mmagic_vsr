import requests
url = 'http://localhost:5000/train'
headers = {'Content-Type': 'application/json'}
data = {
    'cfg_path': 'configs/vrt/vrt_c64n7_8xb1-600k_reds4.py',
    'model_parameters': {}
}
response = requests.post(url, headers=headers, json=data)
print(response.status_code)
print(response.text)