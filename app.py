import pickle

from config import HOST, PORT, DEBUG, MODEL_PATH
from flask import Flask, render_template, request
from utils import to_tensor, generate_attention_map, show_attention
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

    hypothesis = beam_search(model, [tokenized_sent], beam_size=3, max_decoding_time_step=70)[0]
    print("Hypothesis")
    print(hypothesis)

    for i in range(3):
        new_target = [['<sos>'] + hypothesis[i].value + ['<eos>']]
        a_ts = generate_attention_map(model, vocabs, [tokenized_sent], new_target)
        show_attention(tokenized_sent, hypothesis[i].value, 
                        [a[0].detach().cpu().numpy() for a in a_ts[:len(hypothesis[i].value)]], 
                        save_path="static/list_{}.png".format(i))

    result_hypothesis = []
    for idx, hyp in enumerate(hypothesis):
        result_hypothesis.append((idx, " ".join(hyp.value)))

    return render_template("index.html", hypothesis=result_hypothesis, sentence=text_input)



if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)