import requests
import re

def cleaningText(input):
    if type(input) == list:
        final = []
        for text in input:
            text = text.lower()
            # removing Punctuation, Emoji, URL, @
            text = list(text)
            i = 1
            while i < len(text) - 1:
                try:
                    digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
                    j = i
                    if text[i - 1] in digits and text[i + 1] in digits and text[i] == '.':
                        while True:
                            if text[j] in digits or text[j] == '.':
                                text[j] = ''
                            else:
                                i = j
                                break
                            j += 1
                    else:
                        i += 1
                except:
                    i += 1
            text = ''.join(text)
            text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", text)
            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r' ', text)  # no emoji
            text = re.sub(r'http\S+', ' ', text)  # no url
            text = text.replace("@", " ")  # no @
            text = text.replace("#", " ")  # no #
            # remove special characters
            text = re.sub(r'\W', ' ', str(text))
            # remove single characters
            text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
            # remove single characters from the start
            text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
            # Substituting multiple spaces with a single space
            text = re.sub(r'\s+', ' ', text)
            # remove prefixed b
            text = re.sub(r'^b\s+', ' ', text)
            text = text.rstrip()
            text = text.lstrip()
            text = re.sub(' +', ' ', text)
            l1 = text.split()
            if len(l1)<5:
                pass
            else:
                final.append(''.join(text))
        return final
    else:
        text = input
        text = text.lower()
        # removing Punctuation, Emoji, URL, @
        text = list(text)
        i = 1
        while i < len(text) - 1:
            try:
                digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
                j = i
                if text[i - 1] in digits and text[i + 1] in digits and text[i] == '.':
                    while True:
                        if text[j] in digits or text[j] == '.':
                            text[j] = ''
                        else:
                            i = j
                            break
                        j += 1
                else:
                    i += 1
            except:
                i += 1
        text = ''.join(text)
        text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", text)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r' ', text)  # no emoji
        text = re.sub(r'http\S+', ' ', text)  # no url
        text = text.replace("@", " ")  # no @
        text = text.replace("#", " ")  # no #
        # remove special characters
        text = re.sub(r'\W', ' ', str(text))
        # remove single characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        # remove single characters from the start
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
        # Substituting multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # remove prefixed b
        text = re.sub(r'^b\s+', ' ', text)
        text = text.rstrip()
        text = text.lstrip()
        text = re.sub(' +', ' ', text)
        l1 = text.split()
        if len(l1) < 5:
            return None
        else:
            return ''.join(text)

def getNewsAPI(query):
    url = "https://free-news.p.rapidapi.com/v1/search"
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
        news.append(item1[4])
        final.append(news)
    ans = []
    for items in final:
        news = []
        x = items[0].find(':')
        news.append(items[0][x + 2:-1])
        y = items[1].find(':')
        news.append(items[1][y + 2:-1])
        ans.append(news)
    res=[]
    for i in range(len(ans)):
        d = {'title': "", 'link': ""}
        d['title'] = cleaningText(ans[i][0])
        d['link'] = ans[i][1]
        res.append(d)
    res=res[:5]
    return res
print(getNewsAPI('narendra modi'))
