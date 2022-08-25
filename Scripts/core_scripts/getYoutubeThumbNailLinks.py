api_key = 'AIzaSyC7NHT2t9sXCMBKhel1diaIBJ3GaBFt0dk'
from googleapiclient.discovery import build
yt = build('youtube','v3',developerKey=api_key)

keywords = ['government of india','parliament','rbi']
request = yt.search().list(part="snippet",
        maxResults=25,
        q=keywords)

response = request.execute()
image_urls =[]
# print(response)
print("\n\n")
for i in range(25):
    url = response['items'][i]["snippet"]['thumbnails']['high']['url']
    image_urls.append(url)

for l in image_urls:
    print(l)


