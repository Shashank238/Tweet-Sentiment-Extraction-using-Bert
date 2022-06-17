import config
import torch
import flask
import time
import numpy as np
from flask import Flask , request, render_template
from flask_cors import cross_origin
from flask import request
from src.model import TweetModel
import functools
import torch.nn as nn

app = Flask(__name__,template_folder='templates')


DEVICE = config.DEVICE
MODEL = None


def sentence_ext(sentence,sentiment):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    sen = str(sentence)
    sen = " ".join(sen.split())
    senti = sentiment.lower()
    
    tok_tweet = tokenizer.encode(sen)
    input_ids_orig = tok_tweet.ids[1:-1]
    tweet_offsets = tok_tweet.offsets[1:-1]
    
    sentiment_id = {
        'positive': 3893,
        'negative': 4997
    }
    
    input_ids = [101] + [sentiment_id[senti]] + [102] + input_ids_orig + [102]
    token_type_ids = [0, 0, 0] + [1] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 3 + tweet_offsets + [(0, 0)]
    
    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        
    
    ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)
    senti=senti
    original_tweet_sp=sen
    offsets_start = torch.tensor([x for x, _ in tweet_offsets], dtype=torch.long)
    offsets_end = torch.tensor([x for _, x in tweet_offsets], dtype=torch.long)
    
    outputs_start, outputs_end = MODEL(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids,
        )
    outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
    outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy() 
    idx_start=np.argmax(outputs_start)
    idx_end=np.argmax(outputs_end)
    
    offsets = list(zip(offsets_start, offsets_end))
     
    if idx_end < idx_start:
         idx_end = idx_start
    
    filtered_output  = ""
    original_tweet_sp = " ".join(original_tweet_sp.split())
    for ix in range(idx_start, idx_end + 1):
        if offsets[ix][0] == 0 and offsets[ix][1] == 0:
            continue
        filtered_output += original_tweet_sp[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "
            
    filtered_output = filtered_output.replace(" .", ".")
    filtered_output = filtered_output.replace(" ?", "?")
    filtered_output = filtered_output.replace(" !", "!")
    filtered_output = filtered_output.replace(" ,", ",")
    filtered_output = filtered_output.replace(" ' ", "'")
    filtered_output = filtered_output.replace(" n't", "n't")
    filtered_output = filtered_output.replace(" 'm", "'m")
    filtered_output = filtered_output.replace(" do not", " don't")
    filtered_output = filtered_output.replace(" 's", "'s")
    filtered_output = filtered_output.replace(" 've", "'ve")
    filtered_output = filtered_output.replace(" 're", "'re")
    
    return original_tweet_sp,filtered_output

@app.route("/")
@cross_origin()
def home():
    return render_template("predict.html")


@app.route("/", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        
        sentence = request.form['sentence']
        sentiment = request.form['sentiment']
        
        original_tweet_sp,result = sentence_ext(sentence,sentiment)
        
        return render_template("predict.html",sent=original_tweet_sp,prediction=result)
        
        

if __name__ == '__main__':
    MODEL = TweetModel()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE) 
    MODEL.eval()
    app.run(debug=True,port=3000)
    
    
    
    
    
    
    
    
    
    
    
    
        
         
        
        
        
        
        