from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pymorphy2
import requests
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import os
import numpy as np

morph = pymorphy2.MorphAnalyzer(lang='ru')
stop = set(stopwords.words('russian'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def remove_punctuation(text_string: str) -> str:
    return re.sub(r'[^\w\s]', ' ', text_string)

def has_numbers(text_string: str) -> bool:
    return any(char.isdigit() for char in text_string)

def has_ascii(text_string: str) -> bool:
    return any(char.isascii() for char in text_string)

def get_text_from_url(url, encoding='utf-8', to_lower=True):
    url = str(url)
    if url.startswith('http'):
        r = requests.get(url)
        if not r.ok:
            r.raise_for_status()
        return r.text.lower() if to_lower else r.text
    elif os.path.exists(url):
        with open(url, encoding=encoding) as f:
            return f.read().lower() if to_lower else f.read()
    else:
        raise Exception('parameter [url] can be either URL or a filename')

rus_stopwords_url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"
rus_stopwords = get_text_from_url(rus_stopwords_url).splitlines()

print("There are 10 sentences of following two classes on which K-NN classification and K-means clustering"
      " is performed : \n1. Detective \n2. Space Science")
path = "Sentences.txt"

train_clean_sentences = []
fp = open(path, 'r', errors='ignore')
for line in fp:
    token_string = str()
    line = remove_punctuation(line)
    for word in line.split():
        if not any([word.isnumeric(), word.isascii(), has_numbers(word), has_ascii(word)]):
            token_string += ' ' + morph.parse(word.lower())[0].normal_form
    train_clean_sentences.append(token_string)

vectorizer = TfidfVectorizer(stop_words= rus_stopwords)
X = vectorizer.fit_transform(train_clean_sentences)

y_train = np.zeros(20)
y_train[10:20] = 1

modelknn = KNeighborsClassifier(n_neighbors=3)
modelknn.fit(X, y_train)

test_sentences = ["Ее ДНК. Они хотели доказательств, что убитый действительно был ее братом. Кэрри Отто почтой прислала образец слюны, и мы сверили его с анализом жертвы. Оказалось семейное сходство.",
                  "Полтонны разогнанного до высоких скоростей щебня заставили два их боевых корабля сменить курс. Уменьшение добычи воды в кольце Сатурна было вызвано то ли запретом на нелегальные работы, то ли необходимостью усиленной охраны. Две принадлежащие Земле рудные шахты подверглись атаке — Марса или АВП. Погибло четыреста человек, пошел третий месяц блокады Марса, установленной Землей."]

test_clean_sentence = []
for test in test_sentences:
    token_string = str()
    for word in test.split():
        if not any([word.isnumeric(), word.isascii(), has_numbers(word), has_ascii(word)]):
            token_string += ' ' + morph.parse(word.lower())[0].normal_form
    test_clean_sentence.append(token_string)

Test = vectorizer.transform(test_clean_sentence)

true_test_labels = ['Detective', ' Space Science']
predicted_labels_knn = modelknn.predict(Test)

print("\nBelow 2 sentences will be predicted against the learned nieghbourhood and learned clusters:\n1. ", \
      test_sentences[0], "\n2. ", test_sentences[1])
print("\n-------------------------------PREDICTIONS BY KNN------------------------------------------")
print("\n", test_sentences[0], ":", true_test_labels[np.int(predicted_labels_knn[0])], \
      "\n", test_sentences[1], ":", true_test_labels[np.int(predicted_labels_knn[1])])

