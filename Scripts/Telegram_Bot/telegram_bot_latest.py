import requests
import base64
import json
import logging
import requests
from telegram.ext import Updater , CommandHandler , MessageHandler , Filters
from fastai.vision.all import load_learner
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from transformers import pipeline , AutoTokenizer, TFAutoModelForSequenceClassification
from detoxify import Detoxify
import fasttext
import numpy as np
import pickle
from threading import *

APIKey = "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"

def start(update, context):
    update.message.reply_text(
        "EN : Just give me a news and I will tell you whether it is FAKE or not"
    )

def help_command(update, context):
    update.message.reply_text('My only purpose is to tell you whether a given news is fake or not')

def load_model():
    global modelClassify , clickBaitModel , sentimentModel , biasModel , tokenizer , model_classify_second
    model_classify_second = pickle.load(open('./final_model.sav', 'rb'))
    modelClassify = TFAutoModelForSequenceClassification.from_pretrained('ritabrata/pibnews-distilbert')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    clickBaitModel = pipeline(model="elozano/bert-base-cased-clickbait-news", tokenizer="elozano/bert-base-cased-clickbait-news")
    sentimentModel = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    biasModel = pipeline(model="d4data/bias-detection-model", tokenizer="d4data/bias-detection-model")
    print("All models are loaded")

# class load_primary_classification(Thread):
#     def run(self):
#         global modelClassify , tokenizer
#         modelClassify = TFAutoModelForSequenceClassification.from_pretrained('ritabrata/pibnews-distilbert')
#         tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# class load_secondary_classification(Thread):
#     def run(self):
#         global model_classify_second
#         model_classify_second = pickle.load(open('/content/drive/MyDrive/Telegram_bot/final_model.sav', 'rb'))

# class load_feature_models(Thread):
#     def run(self):
#         global clickBaitModel , sentimentModel , biasModel
#         clickBaitModel = pipeline(model="elozano/bert-base-cased-clickbait-news", tokenizer="elozano/bert-base-cased-clickbait-news")
#         sentimentModel = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")
#         biasModel = pipeline(model="d4data/bias-detection-model", tokenizer="d4data/bias-detection-model")

def detect_news(update, context):
    news = update.message.text

    update.message.reply_text("Waiting for the output....")

    textToReply , finalNewsFeatures = prediction(news)

    update.message.reply_text(textToReply)
    
    update.message.reply_text("The news features are: ")
    for key , val in finalNewsFeatures.items():
        update.message.reply_text(key + ": " + val)

def detect_image(update , context):
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')
    img_path = 'user_photo.jpg'
    img_text = preprocess(get_text_from_image(img_path))
    if len(img_text) > 0:
        update.message.reply_text("Waiting for the output...")
        textToReply , finalNewsFeatures = prediction(img_text)
        update.message.reply_text(textToReply)
        update.message.reply_text("The news features are: ")
        for key , val in finalNewsFeatures.items():
            update.message.reply_text(key + ": " + val)        
    else:
        update.message.reply_text("The model was not able to parse text from the given image")

def preprocess(text):
    PRETRAINED_MODEL_PATH = './lid.176.bin'
    model = fasttext.load_model(PRETRAINED_MODEL_PATH)
    return ' '.join([i  for i in text.split(' ') if len(i) != 1 if '__label__en' in model.predict(i, k=3)[0]])

def get_text_from_image(img_path) :
    url = "https://app.nanonets.com/api/v2/OCR/FullText"
    payload={'urls': ['MY_IMAGE_URL']}
    files=[('file',('FILE_NAME',open(img_path,'rb'),'application/pdf'))]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files, auth=requests.auth.HTTPBasicAuth('I-yhRSzNQmxj8dfhXKUQVA55Wj_1Sqjy', ''))

    return json.loads(response.text)['results'][0]['page_data'][0]['raw_text']

def prediction(news):
    finalNewsFeatures = getNewsFeatures(news)
    tokenized = tokenizer([news], return_tensors="np", padding="longest")
    outputs = modelClassify(tokenized).logits
    result = np.argmax(outputs, axis=1)[0]
    if result == 1:
        textToReply = ("The given news is NOT FAKE")
    else:
        print("Going to the second model")
        prediction = model_classify_second.predict([news])
        prob = model_classify_second.predict_proba([news])
        result = prediction[0]
        if result == True:
            textToReply = ("The given news is NOT FAKE")
        else:
            textToReply = ("The given news is FAKE")

    return textToReply , finalNewsFeatures

def getNewsFeatures(inputText) :
    finalNewsFeatures = {}
    results_model1 = clickBaitModel(inputText)[0]
    if(results_model1['label']=='Clickbait') :
        finalNewsFeatures.__setitem__('Clickbait probability', str(round((results_model1['score']*100), 2))+"%")
    else :
        finalNewsFeatures.__setitem__('Clickbait probability', str(round(((1-results_model1['score'])*100), 2))+"%")
    results_model2 = sentimentModel(inputText)[0]
    finalNewsFeatures.__setitem__('Sentiment', results_model2['label'])
    results_model3 = biasModel(inputText)[0]
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



def main():
    load_model()
    # t1 = load_primary_classification()
    # t2 = load_secondary_classification()
    # t3 = load_feature_models()

    # t1.start()
    # t2.start()
    # t3.start()
    
    # t1.join()
    # t2.join()
    # t3.join()

    print("All models are loaded")
    TOKEN = "5477065061:AAFOxVFSTfnrwuSCbsxbUDtpwh3zXhTMk4Q"
    updater = Updater(token = TOKEN , use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(MessageHandler(Filters.text, detect_news))
    dp.add_handler(MessageHandler(Filters.photo , detect_image))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()





