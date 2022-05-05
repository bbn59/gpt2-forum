#!/usr/bin/env python3

import os
import time
import tensorflow as tf
import datetime
from waitress import serve
from transformers import TFGPT2LMHeadModel
from transformers import GPT2Tokenizer
from flask import Flask, render_template, request, redirect, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import DateTime
from sqlalchemy.sql import text
from sqlalchemy.ext.declarative import declarative_base


app = Flask(__name__)
Base = declarative_base()
BASEDIR = os.path.abspath(os.path.dirname(__file__))
SQL_ALCH_URL = "SQLALCHEMY_DATABASE_URI"
app.config[SQL_ALCH_URL] = "sqlite:///" + os.path.join(BASEDIR, 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
DB = SQLAlchemy(app)

tf.config.set_visible_devices([], 'GPU')
tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2")
model = TFGPT2LMHeadModel.from_pretrained("models/gpt2", pad_token_id=tokenizer.eos_token_id)
# Uncomment and use your own
# See: https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling
#model = TFGPT2LMHeadModel.from_pretrained("models/<your custom model>", pad_token_id=tokenizer.eos_token_id)

class content(DB.Model):
    __tablename__ = "posts"
    id = DB.Column(DB.Integer, primary_key=True)
    user_content = DB.Column(DB.String(120), unique=False, index=False)
    gpt_content = DB.Column(DB.String(1048), unique=False, index=False)
    timestamp = DB.Column(
        DB.DateTime, default=datetime.datetime.utcnow, index=True)


DB.drop_all()
DB.create_all()


@app.route('/favicon.ico')
def favicon():
    return ':^)'


@app.route('/', methods=['GET'])
def home():

    posts = content.query.from_statement(text(
        '''
    SELECT * FROM posts
    WHERE timestamp > DateTime('Now', 'utc', '-2 Hour')
    ORDER BY timestamp DESC
    LIMIT 200;
    '''
    ))

    return render_template(
        "index.html", post={
            "posted": request.method == 'POST',
            "posts": posts
        })


@app.route('/post', methods=['POST'])
def post():

    your_post = str(request.form['post'][:36])
    your_post = your_post + ' '

    if len(your_post) >= 8:

        job_est = str(0)

        input_ids = tokenizer.encode(your_post, return_tensors='tf')

        gpt_output = model.generate(
            input_ids, do_sample=True,
            max_length=50, top_k=35,
            temperature=0.75, no_repeat_ngram_size=3
        )

        result = tokenizer.decode(
            gpt_output[0], skip_special_tokens=True
        )

        result = result[len(your_post)+1:]

        new_post = content(
            user_content=str(your_post),
            gpt_content=str(result)
        )

        DB.session.add(new_post)
        DB.session.commit()

        finished = str(
            '<meta http-equiv="refresh" content = "'
            + job_est
            + '; url = /" />'
            + '<link rel="stylesheet" type="text/css" href="/static/css.css">'
            + '<body><center><br><br><h1>Success!</h1></center></body>'
        )
        return finished

    else:
        job_est = str(0)
        failed = str(
            '<meta http-equiv="refresh" content = "'
            + job_est
            + '; url = /" />'
            + '<link rel="stylesheet" type="text/css" href="/static/css.css">'
            + '<body><center><br><br><h1>Fail.</h1></center></body>'
        )
        return failed


if __name__ == '__main__':
    # app.run()
    serve(app, host='127.0.0.1', port=5000, threads=2)
