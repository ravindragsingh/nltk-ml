documents = [
    "I absolutely loved the movie, it was fantastic!",
    "This is the best meal I have ever had.",
    "I feel so happy and satisfied with the service.",
    "The experience was wonderful and unforgettable.",
    "Everything worked perfectly and exceeded my expectations.",
    "I am very disappointed with the quality of this product.",
    "This was a terrible experience and I regret it.",
    "The service was slow and the staff was rude.",
    "I feel frustrated and unhappy with the results.",
    "The product broke after one use, very poor quality.",
    "What a great day, everything went smoothly!",
    "I really appreciate all the help and support.",
    "The team did an excellent job on the project.",
    "I’m बेहद unhappy with how things turned out.",
    "Nothing about this experience was enjoyable.",
    "I am extremely satisfied with my purchase.",
    "This is absolutely awful, I hate it.",
    "The results are amazing and I couldn’t be happier.",
    "I feel terrible about the whole situation.",
    "Such a pleasant and delightful experience overall.",
    "I purchase Iphone and feel it is 8 out of 10",
    "I have laptop which i feel 4 out of 10 "
]

labels = [1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,1,0,1,0,1,1,0]

# Step1 Tokenization 

import nltk
from nltk.tokenize import word_tokenize

tokens = [word_tokenize(doc.lower()) for doc in documents]
print(tokens)

#Remove stop Words

from nltk.corpus import stopwords
stop_words =set(stopwords.words('english'))
filtered_tokens = [[word for word in doc if word.isalpha() and word not in stop_words] for doc in tokens ]
print(filtered_tokens)

#Perform Stemming Porter stemming which is not very strict one to get the base words
from nltk.stem import PorterStemmer
stemmer =PorterStemmer()
stemmed_token = [[stemmer.stem(word) for word in doc] for doc in filtered_tokens]
print(stemmed_token)

#pos tagging 
from nltk import pos_tag
pos_tags = [pos_tag(doc) for doc in stemmed_token]
print(pos_tags)

# Chunking - chunking is grouping of words based on grammer we are going to define 
from nltk.chunk import RegexpParser
grammar = "NP: {<JJ>*<NN>}"
#in this grammar we are creating noun phrase which consists of ADJECTIVE (JJ) and NOUN (NN)
chunk_parser = RegexpParser(grammar)
for doc in pos_tags:
    tree =chunk_parser.parse(doc)
    print(tree)

#Machine Learning 

processed_docs = [' '.join(doc) for doc in stemmed_token]
print(processed_docs)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1,2))
X = tfidf.fit_transform(processed_docs)

#print(tfidf.get_feature_names_out())
print(X.toarray())

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X,labels)

#test_sentence =["This is worst product"]
test_sentence =["I rate 2 out of 10"]
test_tokens = word_tokenize(test_sentence[0].lower())
test_filtered =  [w for w in test_tokens if w.isalpha() and w not in stop_words]
test_stemmed = [stemmer.stem(w) for w in test_filtered]
test_processed = [" ".join(test_stemmed)]
test_vector = tfidf.transform(test_processed)

prediction = model.predict(test_vector)

if prediction[0]==1:
    print("Positive Sentiments")
else:
    print("Negative Sentiments")