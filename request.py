import requests
import json
url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'brand':2.0, 'price':462.39, 'dayofweek':0,'category_code_split1':4.0,'category_code_split2':11.0,'activity_count':1.0,'Time_Spend':143.0})

print(r.json())