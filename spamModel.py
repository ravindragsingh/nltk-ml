spam_sentences = [
    "Congratulations! You’ve been selected to win a free iPhone—click here now to claim your prize.",
    "Act fast! Limited-time offer—get 90% off luxury watches today only.",
    "You have an unclaimed reward waiting—log in immediately to receive it.",
    "Earn $5,000 a week working from home with no experience required!",
    "Your account has been compromised—verify your details here to secure it.",
    "Don’t miss out! This miracle pill will help you lose 20 pounds in a week.",
    "Exclusive deal just for you—buy now and get three items free!",
    "We tried to deliver your package—click this link to reschedule delivery.",
    "Hot singles in your area are waiting—sign up now for free!",
    "This is your final notice—your subscription will be canceled unless you act now."
]

not_spam_sentences = [
    "Hi, just checking if you're available for a meeting tomorrow afternoon.",
    "Please find attached the report from last week's project review.",
    "Can you pick up some groceries on your way home?",
    "Reminder: your dentist appointment is scheduled for Friday at 10 AM.",
    "It was great catching up with you—let's do it again soon.",
    "The weather looks nice this weekend, maybe we can go for a walk.",
    "Your order has been shipped and should arrive within 3–5 business days.",
    "Thanks for your help on the presentation, it made a big difference.",
    "Let me know if you need any assistance with the assignment.",
    "Happy birthday! Hope you have a wonderful day celebrating."
]

#Data Labeling
documents = spam_sentences + not_spam_sentences
labels = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]

#Feature Extraction (Vectorization) convert human language into a grid of numbers (math) that the computer can actually process.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1,2))
X = tfidf.fit_transform(documents)

#print(tfidf.get_feature_names_out())
print(X.toarray())

#Model Training:feed those numbers and labels into a Logistic Regression algorithm to help it learn the patterns of spammy words.

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X,labels)

#new_text =["Congratulations you won the prize"]
new_text =["Reminder for meeting at 11 AM"]
new_vector = tfidf.transform(new_text)
prediction = model.predict(new_vector)
if prediction[0]==1:
    print("spam")
else:
    print("not spam")