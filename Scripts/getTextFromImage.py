import requests
import base64
import json

url = "https://app.nanonets.com/api/v2/OCR/FullText"

payload={'urls': ['MY_IMAGE_URL']}

def get_text(img_path) :

  files=[
    ('file',('FILE_NAME',open(img_path,'rb'),'application/pdf'))
  ]
  headers = {}

  response = requests.request("POST", url, headers=headers, data=payload, files=files, auth=requests.auth.HTTPBasicAuth('I-yhRSzNQmxj8dfhXKUQVA55Wj_1Sqjy', ''))

  return json.loads(response.text)['results'][0]['page_data'][0]['raw_text']

image_path = '/content/font 2.png'

print(get_text(image_path))
