import pandas as pd

#Step 1 : Reading the data
documents = [
    "I love machine learning",
    "machine learning is fun",
    "I love coding"
]
#Step 2 : Tokenization
tokens = [doc.lower().split() for doc in documents]
print(tokens)

#Step 3 : Building the Vocabulary
vocab = sorted(set(word for doc in tokens for word in doc))
print(vocab)

#Step 4: Count words and covert into data frame
words_occurance = []
for doc in tokens:
    word_count = {word: doc.count(word) for word in vocab}
    words_occurance.append(word_count)

df = pd.DataFrame(words_occurance, index=documents)
print(df)

#step 5 : Count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
Y = vectorizer.get_feature_names_out()
print(X.toarray())

#Step 6 : TF IDF Vectorization 
# TF : Measures the frequency of Term t in Document d . Formula count of terms/ total number of terms
# IDF : Inverse Document Freq. Mesuares how rare a term is occuring accross entire corpus formula : log( (1+n)/(1+df(t)) + 1) 
# Vectorization

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(documents)
print(tfidf.get_feature_names_out())
print(X_tfidf.toarray())

# ### from sklearn.feature_extraction.text import TfidfVectorizer
# docs = [
#     "Ram love Data",
#     "Shyam love Science"
# ]
# tfidf = TfidfVectorizer()
# X_tfidf = tfidf.fit_transform(docs)
# print(tfidf.get_feature_names_out())
# print(X_tfidf.toarray())

# from sklearn.feature_extraction.text import TfidfVectorizer

# documents = ["love data science","love machine learning","data science fun" ]
# tfidf = TfidfVectorizer()
# X_tfidf = tfidf.fit_transform(documents)
# print(tfidf.get_feature_names_out())
# print(X_tfidf.toarray())
# ###
