import pickle

from config import HOST, PORT, DEBUG, MODEL_PATH
from flask import Flask, render_template, request
from utils import to_tensor
from models import Seq2Seq
from test import beam_search
from vocab import Vocab
from train import Vocabularies

from nltk.tokenize import RegexpTokenizer

puncts_except_apostrophe = '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'
TOKENIZE_PATTERN = fr"[{puncts_except_apostrophe}]|\w+|['\w]+"
tokenizer = RegexpTokenizer(pattern=TOKENIZE_PATTERN)

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/translate", methods=["GET", "POST"])
def translate():
    
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        args = request.form

    print(args)
    text_input = args["textarea"]

    print("Input: ", text_input)
    tokenized_sent = tokenizer.tokenize(text_input)
    print("Tokenized input: ", tokenized_sent)

    with open("vocabs.pkl", "rb") as f:
        vocabs = pickle.load(f)

    model = Seq2Seq.load(MODEL_PATH)
    model.device = "cpu"
    hypothesis = beam_search(model, [tokenized_sent], beam_size=3, max_decoding_time_step=70)

    hypothesis = hypothesis[0]
    result_hypothesis = []
    for hyp in hypothesis:
        result_hypothesis.append(" ".join(hyp[0]))

    print(result_hypothesis)

    # print(hypothesis)
    # result_hypothesis = [
    #     "123",
    #     "345",
    #     "678"
    # ]
    return render_template("index.html", hypothesis=result_hypothesis, sentence=text_input)



if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)