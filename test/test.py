import requests
import pandas as pd 

# df = pd.read_csv('../taghche_sample.csv')
path = "C:/Users/Hojjat/Python projects/ParsBertApi/taghche_sample.csv"
# http://localhost:5000/predict
resp = requests.post("http://localhost:5000/predict", json={'text': ' اصلا پیشنهاد نمیکنم پولتون رو دور نریزید خیلی بدرد نخور بود'})

print(resp.text)