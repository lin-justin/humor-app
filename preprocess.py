import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import texthero as th

train_data = pd.read_csv("data/train.csv")
# fastText only recognizes labels that are prefixed with "__label__"
train_data.humor = train_data.humor.map({False : "__label__not_humorous",
                                         True : "__label__humorous"})

valid_data = pd.read_csv("data/valid.csv")
valid_data.humor = valid_data.humor.map({False : "__label__not_humorous",
                                         True : "__label__humorous"})

# Preprocessing/cleaning the text
# Reference: https://texthero.org/docs/getting-started
# ===============================
# Replace not assigned values with empty spaces
# Lowercase all text
# Remove all blocks or digits
# Remove all string punctuation
# Remove all accents
# Remove all stopwords
# Remove all white space between words
train_data.text = th.clean(train_data.text)
valid_data.text = th.clean(valid_data.text)

# Save it as tab-separated text file for fastText to recognize
train_data.to_csv("data/train_clean.txt", index = False, header = False, sep = "\t")
valid_data.to_csv("data/valid_clean.txt", index = False, header = False, sep = "\t")