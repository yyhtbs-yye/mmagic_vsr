import requests
url = 'http://localhost:5000/train'
headers = {'Content-Type': 'application/json'}
data = {
    'cfg_path': 'configs/pcdnet/pcdnet_c64n7_8xb1-600k_reds4.py',
    'model_parameters': {
        'pyramid_depth': "ResiduePCD"
    }
}
response = requests.post(url, headers=headers, json=data)
print(response.status_code)
print(response.text)

# data = {
#     'cfg_path': 'configs/pcdnet/pcdnet_c64n7_8xb1-600k_reds4.py',
#     'model_parameters': {
#         'pyramid_depth': 2
#     }
# }
# response = requests.post(url, headers=headers, json=data)
# print(response.status_code)
# print(response.text)

# data = {
#     'cfg_path': 'configs/pcdnet/pcdnet_c64n7_8xb1-600k_reds4.py',
#     'model_parameters': {
#         'pyramid_depth': 1
#     }
# }

# response = requests.post(url, headers=headers, json=data)
# print(response.status_code)
# print(response.text)
