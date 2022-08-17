#pip install detoxify

from transformers import pipeline

def loadModels() :
    model1 = pipeline(model="elozano/bert-base-cased-clickbait-news", tokenizer="elozano/bert-base-cased-clickbait-news")
    model2 = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    model3 = pipeline(model="d4data/bias-detection-model", tokenizer="d4data/bias-detection-model")
    from detoxify import Detoxify
    return (model1,model2,model3)
    
models = loadModels()
model1 = models[0]
model2 = models[1]
model3 = models[2]

def getNewsFeatures(inputText) :
    finalNewsFeatures = {}
    results_model1 = model1(inputText)[0]
    if(results_model1['label']=='Clickbait') :
        finalNewsFeatures.__setitem__('Clickbait probability', str(round((results_model1['score']*100), 2))+"%")
    else :
        finalNewsFeatures.__setitem__('Clickbait probability', str(round(((1-results_model1['score'])*100), 2))+"%")
    results_model2 = model2(inputText)[0]
    finalNewsFeatures.__setitem__('Sentiment', results_model2['label'])
    results_model3 = model3(inputText)[0]
    if(results_model3['label']=='Biased') :
        finalNewsFeatures.__setitem__('Biased percentage', str(round((results_model3['score']*100), 2))+"%")
    else :
        finalNewsFeatures.__setitem__('Biased percentage', str(round(((1-results_model3['score'])*100), 2))+"%")
    results_model4 = Detoxify('original').predict(inputText)
    finalNewsFeatures.__setitem__('Toxicity percentage', str(round((results_model4['toxicity']*100), 2))+"%")
    finalNewsFeatures.__setitem__('Obscene percentage', str(round((results_model4['obscene']*100), 2))+"%")
    finalNewsFeatures.__setitem__('Insult percentage', str(round((results_model4['insult']*100), 2))+"%")
    finalNewsFeatures.__setitem__('Hatred percentage', str(round((results_model4['identity_attack']*100), 2))+"%")
    finalNewsFeatures.__setitem__('Threat percentage', str(round((results_model4['threat']*100), 2))+"%")
    return finalNewsFeatures
