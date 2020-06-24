import sys
import os
import re

import random
import pandas as pd
import nltk

# replace apostrophe/short words in python
from contractions import contractions_dict

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))

from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

# database
# sys.path.append(os.path.abspath('../../Database'))
import Database.database_log as database_log


if __name__ == "__main__":
    pass
