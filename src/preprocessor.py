import nltk

nltk.data.path.append('./nltk_data')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

pattern = r"""(?x)                       # set flag to allow verbose regexps 
                  (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
                  |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
                """


class Preprocessor(object):

    # turn the text to lower case
    def text_lowercase(self, text):
        
        return text.lower()

    # conduct text segmentation
    def text_segmentation(self, text):
        
        return nltk.regexp_tokenize(text, pattern)

    # remove stop words from text
    def stopword_removal(self, text):
        
        stopwordset = set(stopwords.words('english'))
        return [word for word in text if word not in stopwordset]

    # lemmatize the text
    def lemmatization(self, text):
        
        lemma = WordNetLemmatizer()
        return [lemma.lemmatize(word, 'v') for word in text]
