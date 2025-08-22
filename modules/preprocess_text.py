# ========== Import Libraries =========
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# load stop words
stopwords = set(stopwords.words('english'))

# Instantiate the lemmatizer
lemmarizer = WordNetLemmatizer()


# ======== Tokenizer Function ========
def tokenize_text(text):
    """This function tokenizes the input text, removes stop words, and lemmatizes the tokens.
    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of lemmatized tokens."""
    
    
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stopwords]
    lemmatized_tokens = [lemmarizer.lemmatize(word) for word in filtered_tokens if word.isalpha()]
    return lemmatized_tokens



# ======== Vectorizer Function ========
def build_vectorizer(corpus):
    """Builds a TF-IDF vectorizer from the given corpus.
    Args:
        corpus (list): A list of text documents.
    Returns:
        tuple: A tuple containing the TF-IDF matrix, feature names, and a DataFrame representation.
    """

    vectorizer = TfidfVectorizer(tokenizer=tokenize_text)
    X = vectorizer.fit_transform(corpus)

    values = X.toarray()
    feature_names = vectorizer.get_feature_names_out()
    dataframe = pd.DataFrame(values, columns=feature_names)
    return values, feature_names, dataframe