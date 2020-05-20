import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

def text_prep(X):
    documents = []

    from nltk.stem import WordNetLemmatizer
    from nltk.stem import PorterStemmer

    lemmatizer = WordNetLemmatizer()
    porter = PorterStemmer()

    for sen in range(0, len(X)):
    # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
    
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
    
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
    
        # Converting to Lowercase
        document = document.lower()

        #remove stopwords
        stop_words = set(stopwords.words("english")) 
        word_tokens = word_tokenize(document)  
        document = [word for word in word_tokens if not word in stop_words] 
        document = ' '.join(document)
    
        # Lemmatization
        word_tokens = word_tokenize(document)
        document = [porter.stem(word) for word in word_tokens]
        document = ' '.join(document)
    
        documents.append(document)
    return documents
