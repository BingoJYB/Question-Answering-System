from preprocessing import Preprocessing
from data import answers


preprocessor = Preprocessing()
text = answers['method 1']['step 1']
print(preprocessor.stopword_removal(preprocessor.text_segmentation(text)))
