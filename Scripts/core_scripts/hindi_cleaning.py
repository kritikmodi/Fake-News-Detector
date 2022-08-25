# -*- coding: utf-8 -*-
"""Hindi Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Fb9BSpWMSw1TuOkN-Er8jeKdQblel1y5
"""

# !pip install fasttext
# import fasttext
# !wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    
# PRETRAINED_MODEL_PATH = '/tmp/lid.176.bin'
# model = fasttext.load_model(PRETRAINED_MODEL_PATH)

# def hindi_extract(text):
#     return ' '.join([i  for i in text.split(' ') if '__label__hi' in model.predict(i, k=3)[0]])

import re
import codecs,string
from numpy import unicode
def is_hindi(input):
    l=[]
    for character in input:
      maxchar = max(character)
      digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
      if u'\u0900' <= maxchar <= u'\u097f' or character in digits:
        l.append(character)
    return ''.join(l)
def hindi_extract(input):
    if type(input)==list:
        final = []
        for text in input:
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
            # text = ' '.join([i  for i in text.split(' ') if '__label__hi' in model.predict(i, k=3)[0]])
            text = re.sub(r"(\w+:\/\/\S+)|^rt|http.+?", " ", text)
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
            text = re.sub(r'^b\s+', ' ', text)
            text = text.rstrip()
            text = text.lstrip()
            text = re.sub(' +', ' ', text)
            new_str = ""
            for items in text.split():
              res = is_hindi(items)
              if res:
                new_str+=res+" "
            l1 = new_str.split()
            if len(l1)<5:
                pass
            else:
                final.append(' '.join(l1))
        return final
    else:
        text = input
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
        # text = ' '.join([i  for i in text.split(' ') if '__label__hi' in model.predict(i, k=3)[0]])
        text = re.sub(r"(\w+:\/\/\S+)|^rt|http.+?", " ", text)
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
        text = re.sub(r'^b\s+', ' ', text)
        text = text.rstrip()
        text = text.lstrip()
        text = re.sub(' +', ' ', text)
        new_str = ""
        for items in text.split():
          res = is_hindi(items)
          if res:
            new_str+=res+" "
        l1 = new_str.split()
        if len(l1) < 5:
            return None
        else:
            return ' '.join(l1)

suffixes = {
    1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],
    2: ["कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती", "ता", "ाँ", "ां", "ों", "ें"],
    3: ["ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं", "ुआं"],
    4: ["ाएगी", "ाएगा", "ाओगी", "ाओगे", "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं", "ताएं", "ियाँ", "ियों", "ियां"],
    5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां"],
}

def hi_stem(word):
    for L in 5, 4, 3, 2, 1:
        if len(word) > L + 1:
            for suf in suffixes[L]:
                if word.endswith(suf):
                    return word[:-L]
    return word

input = ["मीडिय आउटलेट्स और वायर एजेंस ने गलत तरीक से दाव किय कि पाकिस्तान के पूर्व राजनयिक ने बालाकोट में हुई मौत की",
         "बूम ने पाय कि इमरान खान भारत सरकार की आलोच कर रह थे और वर्तमान शासन को अधिनायकवाद मुस्लिम विरोध और पाकिस्तान"]
input = hindi_extract(input)
def getStem(input):
  if type(input)==list:
    final =[]
    str_temp=""
    for sen in input:
      for words in sen.split():
        str_temp+=hi_stem(words)
        str_temp+=" "
      final.append(str_temp)
      str_temp=""
    return final
  else:
    str_temp=""
    for words in input.split():
      str_temp+=hi_stem(words)
      str_temp+=" "
    return str_temp
print(getStem(input))

stopwords=['मैं',
 'मुझको',
 'मेरा',
 'अपने आप को',
 'हमने',
 'हमारा',
 'अपना',
 'हम',
 'आप',
 'आपका',
 'तुम्हारा',
 'अपने आप',
 'स्वयं',
 'वह',
 'इसे',
 'उसके',
 'खुद को',
 'कि वह',
 'उसकी',
 'उसका',
 'खुद ही',
 'यह',
 'इसके',
 'उन्होने',
 'अपने',
 'क्या',
 'जो',
 'किसे',
 'किसको',
 'कि',
 'ये',
 'हूँ',
 'होता है',
 'रहे',
 'थी',
 'थे',
 'होना',
 'गया',
 'किया जा रहा है',
 'किया है',
 'है',
 'पडा',
 'होने',
 'करना',
 'करता है',
 'किया',
 'रही',
 'एक',
 'लेकिन',
 'अगर',
 'या',
 'क्यूंकि',
 'जैसा',
 'जब तक',
 'जबकि',
 'की',
 'पर',
 'द्वारा',
 'के लिए',
 'साथ',
 'के बारे में',
 'खिलाफ',
 'बीच',
 'में',
 'के माध्यम से',
 'दौरान',
 'से पहले',
 'के बाद',
 'ऊपर',
 'नीचे',
 'को',
 'से',
 'तक',
 'से नीचे',
 'करने में',
 'निकल',
 'बंद',
 'से अधिक',
 'तहत',
 'दुबारा',
 'आगे',
 'फिर',
 'एक बार',
 'यहाँ',
 'वहाँ',
 'कब',
 'कहाँ',
 'क्यों',
 'कैसे',
 'सारे',
 'किसी',
 'दोनो',
 'प्रत्येक',
 'ज्यादा',
 'अधिकांश',
 'अन्य',
 'में कुछ',
 'ऐसा',
 'में कोई',
 'मात्र',
 'खुद',
 'समान',
 'इसलिए',
 'बहुत',
 'सकता',
 'जायेंगे',
 'जरा',
 'चाहिए',
 'अभी',
 'और',
 'कर दिया',
 'रखें',
 'का',
 'हैं',
 'इस',
 'होता',
 'करने',
 'ने',
 'बनी',
 'तो',
 'ही',
 'हो',
 'इसका',
 'था',
 'हुआ',
 'वाले',
 'बाद',
 'लिए',
 'सकते',
 'इसमें',
 'दो',
 'वे',
 'करते',
 'कहा',
 'वर्ग',
 'कई',
 'करें',
 'होती',
 'अपनी',
 'उनके',
 'यदि',
 'हुई',
 'जा',
 'कहते',
 'जब',
 'होते',
 'कोई',
 'हुए',
 'व',
 'जैसे',
 'सभी',
 'करता',
 'उनकी',
 'तरह',
 'उस',
 'आदि',
 'इसकी',
 'उनका',
 'इसी',
 'पे',
 'तथा',
 'भी',
 'परंतु',
 'इन',
 'कम',
 'दूर',
 'पूरे',
 'गये',
 'तुम',
 'मै',
 'यहां',
 'हुये',
 'कभी',
 'अथवा',
 'गयी',
 'प्रति',
 'जाता',
 'इन्हें',
 'गई',
 'अब',
 'जिसमें',
 'लिया',
 'बड़ा',
 'जाती',
 'तब',
 'उसे',
 'जाते',
 'लेकर',
 'बड़े',
 'दूसरे',
 'जाने',
 'बाहर',
 'स्थान',
 'उन्हें ',
 'गए',
 'ऐसे',
 'जिससे',
 'समय',
 'दोनों',
 'किए',
 'रहती',
 'इनके',
 'इनका',
 'इनकी',
 'सकती',
 'आज',
 'कल',
 'जिन्हें',
 'जिन्हों',
 'तिन्हें',
 'तिन्हों',
 'किन्हों',
 'किन्हें',
 'इत्यादि',
 'इन्हों',
 'उन्हों',
 'बिलकुल',
 'निहायत',
 'इन्हीं',
 'उन्हीं',
 'जितना',
 'दूसरा',
 'कितना',
 'साबुत',
 'वग़ैरह',
 'कौनसा',
 'लिये',
 'दिया',
 'जिसे',
 'तिसे',
 'काफ़ी',
 'पहले',
 'बाला',
 'मानो',
 'अंदर',
 'भीतर',
 'पूरा',
 'सारा',
 'उनको',
 'वहीं',
 'जहाँ',
 'जीधर',
 '\ufeffके',
 'एवं',
 'कुछ',
 'कुल',
 'रहा',
 'जिस',
 'जिन',
 'तिस',
 'तिन',
 'कौन',
 'किस',
 'संग',
 'यही',
 'बही',
 'उसी',
 'मगर',
 'कर',
 'मे',
 'एस',
 'उन',
 'सो',
 'अत']

input = getStem(input)
def getstopwords(input):
  if type(input)==list:
    final =[]
    str_temp=""
    for sen in input:
      for words in sen.split():
        if unicode(words) not in stopwords:
          str_temp+=words
          str_temp+=" "
      final.append(str_temp)
      str_temp=""
    return final
  else:
    str_temp=""
    for words in input.split():
        if unicode(words) not in stopwords:
          str_temp+=words
          str_temp+=" "
    return str_temp
print(getstopwords(input))

