import re
import nltk


from nltk.corpus import stopwords


pattern = r"""(?x)                       # set flag to allow verbose regexps 
                  (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A. 
                  |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages 
                  |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
                """ 


class Preprocessing(object):

    def text_segmentation(self, text):
        
        return nltk.regexp_tokenize(text, pattern)
        
    def stopword_removal(self, text):
        
        stopwordset = set(stopwords.words('english'))
        return [word for word in text if word not in stopwordset]
        