import requests

url = "https://free-news.p.rapidapi.com/v1/search"
query = "Narendra Modi"
querystring = {"q": query, "lang": "en"}
headers = {
    "X-RapidAPI-Key": "555298ffecmsh3d72743b2b5fb02p197c87jsnf01abdd5c1fc",
    "X-RapidAPI-Host": "free-news.p.rapidapi.com"
}
response = requests.request("GET", url, headers=headers, params=querystring)
string = response.text
i = string.find('"articles":[{')
i += 11
string = string[i + 2::]
j = string.find("]")
string = string[:j]
docs = string.split('},{')
articles = []
for items in docs:
    news = items.split(',"')
    articles.append(news)
final = []
for item1 in articles:
    news = []
    news.append(item1[0])
    news.append(item1[6])
    final.append(news)
ans = []
for items in final:
    news = []
    x = items[0].find(':')
    news.append(items[0][x + 2:-1])
    y = items[1].find(':')
    news.append(items[1][y + 2:-1])
    ans.append(news)
print(ans)
