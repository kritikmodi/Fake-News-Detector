!pip install python-telegram-bot --upgrade
!pip install tensorflow-text
!pip install transformers
!pip install fasttext
!pip install scipy
!pip3 install "scikit_learn==0.22.2.post1"
!pip install tweepy==4.5.0
!pip install sentence-transformers
!pip install wikipedia

import requests
import base64
import json
import logging
import pandas as pd
from telegram.ext import Updater , CommandHandler , MessageHandler , Filters
from fastai.vision.all import load_learner
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from transformers import pipeline , AutoTokenizer, TFAutoModelForSequenceClassification
import fasttext
import numpy as np
import pickle
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from pprint import pprint
import wikipedia as wiki
from nltk.util import ngrams
import gensim
import re
import nltk
import torch
from nltk import tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import tweepy as tweepy

# nltk.download('stopwords')
# nltk.download('omw-1.4')
nltk.download('punkt')
# nltk.download('wordnet')

APIKey = "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"
!wget -O ./lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
CONSUMER_KEY = "eJL1xOgPnXVx0DzCr5pGa8lNv"
CONSUMER_SECRET = "iDBuPdCEXZQDzsRqvNtkVcIhcdvlT8x8aW74VTm1EqXcIaPmrZ"
OAUTH_TOKEN = "1420293020080082948-Qo8PBaf5oXA1xrPryabo3C3g09xdBf"
OAUTH_TOKEN_SECRET = "HzW0KllX3NRls7pOKA0cPIbNEyAFWON9wgVODcRwlrVBi"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAABKRewEAAAAAem8kRGNx2U5tGTmD%2BtikqmENETI%3DCqzEWLXGEmM8WKNXRRnW0Tke4QlWw2sihgwtjpVYwnAR0QD6bo"
twitterAPI = tweepy.Client(bearer_token = BEARER_TOKEN)


def load_models() :
    global clickBaitModel , sentimentModel , biasModel , classification_after_embedding_model, cae_model_tokenizer, liar_classification_model, wikipedia_model, nlp
    classification_after_embedding_model = TFAutoModelForSequenceClassification.from_pretrained('pururaj/Test_model')
    cae_model_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    clickBaitModel = pipeline(model="elozano/bert-base-cased-clickbait-news", tokenizer="elozano/bert-base-cased-clickbait-news")
    sentimentModel = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    biasModel = pipeline(model="d4data/bias-detection-model", tokenizer="d4data/bias-detection-model")
    liar_classification_model = pickle.load(open('../input/inputdata/liar_classification.sav', 'rb'))
    wikipedia_model = SentenceTransformer('bert-base-nli-mean-tokens')
    nlp = en_core_web_sm.load()

# def load_model_second() :
#     global wikipediaModel , stop_words , lemma
#     model_name = 'flax-sentence-embeddings/all_datasets_v3_roberta-large'
#     wikipediaModel = SentenceTransformer(model_name)
#     wikipediaModel.to('cuda:0' if torch.cuda.is_available() else 'cpu')
#     stop_words = set(stopwords.words('english'))
#     lemma = WordNetLemmatizer()
    
# def load_model_third() :
#     global embeddings , liar_classification
#     embFinal = '../input/inputdata/emb_final.csv'
#     df = pd.read_csv(embFinal)
#     embeddings = df.values.tolist()
#     liar_classification = pickle.load(open('../input/inputdata/liar_classification.sav', 'rb'))

def keywords(sentence):

    doc = nlp(sentence)

    results = ([(X.text, X.label_)[0] for X in doc.ents])
    return list(set(results))

def clean_wiki_text(text):
    text = re.sub(r'==.+==', '.', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\[[0-9]+\]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\. \.', '.', text)
    return text

def content(claim, results):
    sentences = []
    found=[]
    for i in results:
        try :
            current_page = wiki.page(i)
            if current_page not in found:
                found.append(current_page)
        except :
            continue
    titles=[i.title for i in found]
    titles=[i[0] for i in topNSimilar(claim, titles)]
    for i in found:
        if i.title not in titles:
            found.remove(i)

    for i in found:
        current_content = i.content
        sentences.extend(tokenize.sent_tokenize(clean_wiki_text(current_content)))
    return sentences

def topNSimilar(claim, sentList, n=5):
    distList=[]
    for i in sentList:
        document_term_matrix = TfidfVectorizer().fit_transform([i, claim])
        dist = document_term_matrix * document_term_matrix.transpose()
        distance = dist.toarray()[0][1]

        if len(distList)<=n:
            distList.append([i, distance])
        else:
            distList=sorted(distList, key=lambda x: x[1], reverse=True)
            if distance>distList[-1][1]:
                distList.pop()
                distList.append([i, distance])
    return sorted(distList, key=lambda x: x[1], reverse=True)

def topNBert(claim, res):
    topNFacts=[i[0] for i in res]
    topNScore=[cosine_similarity( [wikipedia_model.encode(claim)], [wikipedia_model.encode(i)] )[0][0] for i in topNFacts]
    topN=zip(topNFacts, topNScore)
    return [list(i) for i in list(topN)]

def get_tweet(link) :
    id = int(link.split('/')[-1])
    tweetContent = twitterAPI.get_tweet(id)
    tweetContent = str(tweetContent[0]).split()
    return (" ".join(tweetContent))

# def prep(rowitem) :
#     if len(str(rowitem).split()) < 10:
#         return None
#     rowitem = nltk.tokenize.word_tokenize(rowitem)
#     rowitem = [i.lower() for i in rowitem if i.isalpha()]
#     rowitem = [ i for i in rowitem if i not in stop_words ]
#     rowitem = ' '.join([ lemma.lemmatize(i) for i in rowitem ])
#     return rowitem

# def sim(text, embeddings) :
#     f = 0
#     t = [prep(text)]
#     if t[0] == None:
#         return 'Too small'
#     sen_embeddings = wikipediaModel.encode(t)
# #     print('sen embedded')
#     for idx, i in enumerate(embeddings) :
#         sim = cosine_similarity(list(np.asarray(i).reshape(1, -1)),list(sen_embeddings))
#         if sim > 0.8:
#             f = 1
#             return ('Present '+idx)
#     if not f:
#         return 'Not Present'

def detect_news(news) :

    # if it's a link , treat it as tweet url

    textToReply = prediction(news)
    finalNewsFeatures = getNewsFeatures(news)
    finalWikipediaResults = topNBert(news, topNSimilar(news, content(news, keywords(news))))
    kws=keywords(news)
    cnt=content(news,kws)
    tns=topNSimilar(news,cnt)
    tnb=topNBert(news,tns)
    
    res = []
    res.append(textToReply)
    
    for key , val in finalNewsFeatures.items():
        res.append(val)
        
    for i in tnb :
        res.append(i[0])

def detect_image(update , context) :
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')
    img_path = 'user_photo.jpg'
    img_text = preprocess(get_text_from_image(img_path))
    if len(img_text) > 0:
        update.message.reply_text("Waiting for the output...")
        textToReply = prediction(img_text)
        finalNewsFeatures = getNewsFeatures(news)
        update.message.reply_text(textToReply)
        update.message.reply_text("The news features are: ")
        for key , val in finalNewsFeatures.items():
            update.message.reply_text(key + ": " + val)        
    else:
        update.message.reply_text("The model was not able to parse text from the given image")

def preprocess(text) :
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

def prediction(news) :
    sentences=[news]
    tokenized = cae_model_tokenizer(sentences, return_tensors="np", padding="longest")
    outputs = classification_after_embedding_model(tokenized).logits
    classifications = np.argmax(outputs, axis=1)
    if classifications[0]==0 :
        textToReply = "FAKE"
    else :
        textToReply = "NOT FAKE"
    return textToReply

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
    return finalNewsFeatures

