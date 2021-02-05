import warnings
warnings.filterwarnings("ignore")

import fasttext
import pandas as pd 

# fastText model for text classification
# Reference: https://fasttext.cc/docs/en/supervised-tutorial.html
model = fasttext.train_supervised("data/train_clean.txt", lr = 0.3, epoch = 25, wordNgrams = 2)

model.test("data/valid_clean.txt", k = 1)

test_data = pd.read_csv("data/test.csv")

# model.predict(test_data.text[0])

model.save_model("model_humor.bin")