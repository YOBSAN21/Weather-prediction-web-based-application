import requests

url = 'http://localhost:5000/predict_api'
r1 = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})
r2=requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})
r3 = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})
r4 = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})
r5 = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})
print(r1.json())