import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle as cPickle


# Define the directory containing your text files
mainFolderPath = "Data/Training"

# Initialize a feature extractor (e.g., TF-IDF Vectorizer)
vectorizer = TfidfVectorizer()

# List to store extracted features for each file
mailFeatures = []
mailCategory = []

# Function to extract summary of files
def getSummary(text, num_sentences=3):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)

    # Calculate word frequency
    word_frequencies = {}
    for word in words:
        word = word.lower()
        if word not in stop_words and word.isalpha():
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Score sentences based on word frequency
    sentences = sent_tokenize(text)
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]

    # Select top sentences
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = ' '.join(summary_sentences)
    return summary



# Loop through each file in the directory
for folderName in os.listdir(mainFolderPath):
    #print(folderName)
    subFolder = mainFolderPath + "/" + folderName

    for fileName in os.listdir(subFolder):
            if fileName.endswith(".txt"):  # Process only text files
                filepath = os.path.join(subFolder, fileName)
                with open(filepath, 'r', encoding='utf-8') as file:
                    # Get full content
                    txtContent = file.read()
                    # Get summary of full content
                    txtContent = getSummary(txtContent, num_sentences=3)

                # Perform feature extraction on the text content/mail content
                # Here, we're are performing a per-file operation
                # to get TF-IDF features against the summary of each mail (stored in text file):

                #txtFeature = vectorizer.fit_transform([txtContent]) # Fit and transform on single document
                #print(txtContent)
                # Store the features and Mail Category
                mailFeatures.append(txtContent)
                mailCategory.append(folderName)


data = {'mailFeature': mailFeatures, 'mailCategory':mailCategory}

df = pd.DataFrame(data)

#print(df)

# 2. Feature Extraction (TF-IDF)
# TF-IDF converts text into numerical feature vectors
tfidf_vectorizer = TfidfVectorizer(encoding='utf-8', lowercase=True, analyzer='word')
X = tfidf_vectorizer.fit_transform(df['mailFeature'])
#print(df['mailCategory'])
y = df['mailCategory']

# 3. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Model Training (Multinomial Naive Bayes)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# 5. Prediction and Evaluation
y_pred = classifier.predict(X_test)

#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example of predicting new text
new_text = ["Hi! Saw Panchayat last night. The character of Vinod was outstanding. This web-series was truly amazing! You must watch it"]
new_text_vectorized = tfidf_vectorizer.transform(new_text)
prediction = classifier.predict(new_text_vectorized)
print(f"\nPrediction for Email with content : '{new_text[0]}':\n The e-mail goes to folder: {prediction[0]}")


f = open("classifier.cPickle", "wb")
f.write(cPickle.dumps(classifier))
f.close()