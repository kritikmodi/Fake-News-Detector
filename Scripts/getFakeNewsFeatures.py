import requests

APIKey = "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"
userInput = "jasndjnd"

def getNewsFeatures(userInput, APIKey):
    
    API_URL1 = "https://api-inference.huggingface.co/models/elozano/bert-base-cased-clickbait-news"
    API_URL2 = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"
    API_URL3 = "https://api-inference.huggingface.co/models/d4data/bias-detection-model"
    API_URL4 = "https://api-inference.huggingface.co/models/unitary/toxic-bert"
    headers = {"Authorization": "Bearer "+APIKey}
    
    def query(payload, API_URL):
	    response = requests.post(API_URL, headers=headers, json=payload)
	    return response.json()
    
    output1 = query(userInput, API_URL1)
    output2 = query(userInput, API_URL2)
    output3 = query(userInput, API_URL3)
    output4 = query(userInput, API_URL4)
    
    finalOutput=[]
    finalOutput.append(output1)
    finalOutput.append(output2)
    finalOutput.append(output3)
    finalOutput.append(output4)
    return finalOutput

featuresList = getNewsFeatures(userInput, APIKey)
mydict = {}
for i in featuresList:
    for j in i[0]:
        mydict.__setitem__(j['label'], j['score'])

finalNewsFeatures = {}
finalNewsFeatures.__setitem__('Clickbait probability', str(round((mydict['Clickbait']*100), 2))+"%")
if(mydict['Positive']>=mydict['Negative']>=mydict['Neutral']):
    finalNewsFeatures.__setitem__('Sentiment', 'Positive')
elif(mydict['Negative']>=mydict['Positive']>=mydict['Neutral']):
    finalNewsFeatures.__setitem__('Sentiment', 'Negative')
else:
    finalNewsFeatures.__setitem__('Sentiment', 'Neutral')
finalNewsFeatures.__setitem__('Biased percentage', str(round((mydict['Biased']*100), 2))+"%")
finalNewsFeatures.__setitem__('Toxicity level', str(round((mydict['toxic']*100), 2))+"%")
finalNewsFeatures.__setitem__('Obscene percentage', str(round((mydict['obscene']*100), 2))+"%")
finalNewsFeatures.__setitem__('Insult percentage', str(round((mydict['insult']*100), 2))+"%")
finalNewsFeatures.__setitem__('Hatred percentage', str(round((mydict['identity_hate']*100), 2))+"%")
finalNewsFeatures.__setitem__('Threat percentage', str(round((mydict['threat']*100), 2))+"%")

print(finalNewsFeatures)
