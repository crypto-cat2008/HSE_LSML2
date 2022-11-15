from crypt import methods
from urllib import response
import numpy as np
import pandas as pd
import re
import json
import os

import torch
from transformers import BertTokenizer, BertForSequenceClassification

from flask import Flask, request, render_template

app = Flask(__name__)

content = os.listdir('.')
print("There are {} elements in this directory".format(len(content)))
for element in content:
    print(element)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model1 = BertForSequenceClassification.from_pretrained('./1111model1')
tokenizer1 = tokenizer.from_pretrained('/myapp/1111model1')
model1.to('cpu')

@app.route('/')
def hello():
    return render_template('home.html')

def preprocess(txt):
    regex_string = r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    output=txt
    output=re.sub(regex_string,'',output)
    output=re.sub('(@[^\s]*)\s|\s(@[^\s]*)','',output)
    output=re.sub('(#[^\s]*)\s|\s(#[^\s]*)','',output)
    output=re.sub('[\s]{2}','',output)
    return output

@app.route('/predict', methods=["GET", "POST"])
def model_predict():
    if request.method == 'POST':

        msg=request.form['msg']

        input_ids = []
        attention_masks = []

        encoded_dict = tokenizer1.encode_plus(
                        msg,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        truncation=True,
                        max_length = 350,           # Pad & truncate all sentences.
                        padding='max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )

        input_ids.append(encoded_dict['input_ids'])
    
        attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model1(input_ids, token_type_ids=None, 
                            attention_mask=attention_masks)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Store predictions and true labels
            predictions= logits

        #prediction = str(np.argmax(predictions))

        if np.argmax(predictions) == 1:
            prediction = 'Positive :)'
        else:
            prediction = 'Negative :('


        response = {
                    "result" : str(np.argmax(predictions))
                }
        #return json.dumps(response)
        return render_template('home.html', msg= msg, prediction=prediction)
    else:
        return "You should only use POST query"
