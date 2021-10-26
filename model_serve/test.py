import torch
import requests
import json
url = 'http://localhost:9090/predictions/aminer_RS_rank'
data = json.dumps([1])
re = requests.post(url, data)
a = json.loads(re.text)
