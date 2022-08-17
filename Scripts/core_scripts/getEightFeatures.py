import requests

APIKey = "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"
userInput = "CSGO is made by the company which is named as Valve"

def query(payload, API_URL , APIKey):
    headers = {"Authorization": "Bearer "+APIKey}
    response = requests.post(API_URL, headers=headers, json=payload)

    while "error" in response.json():
        response = requests.post(API_URL, headers=headers, json=payload)

    return response.json()

def getNewsFeatures(userInput, APIKey):
    
    API_URL1 = "https://api-inference.huggingface.co/models/elozano/bert-base-cased-clickbait-news"
    API_URL2 = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"
    API_URL3 = "https://api-inference.huggingface.co/models/d4data/bias-detection-model"
    API_URL4 = "https://api-inference.huggingface.co/models/unitary/toxic-bert"
    
    
    output1 = query(userInput, API_URL1 , APIKey)
    output2 = query(userInput, API_URL2 , APIKey)
    output3 = query(userInput, API_URL3 , APIKey)
    output4 = query(userInput, API_URL4 , APIKey)
    
    finalOutput=[]
    finalOutput.append(output1)
    finalOutput.append(output2)
    finalOutput.append(output3)
    finalOutput.append(output4)
    return finalOutput


def getOutput():
    print("Waiting for the output....")
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


def main():
    getOutput()

if __name__ == '__main__':
    main()

