from flask import Flask, render_template, request
import numpy as np
import LinRegLearner as lrl
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bl

app = Flask(__name__)
app.secret_key = "testing"

LEARNERS_LIST = [
    lrl.LinRegLearner(),
    dtl.DTLearner(leaf_size=3),
    bl.BagLearner(learner=rtl.RTLearner)
]

@app.route("/", methods=["get", "post"])
def index():
    return render_template("landing.html", results=[])

@app.route("/generate/", methods=["post"])
def generate():
    data_text = request.form['data-entry-box']
    data_rows = [row for row in data_text.split("\r\n")]
    data = np.array([[float(n) for n in row.split(",")] for row in data_rows])

    data_x = data[:, :-1]
    data_y = data[:, -1]

    for i in range(len(LEARNERS_LIST)):
        LEARNERS_LIST[i].add_evidence(data_x, data_y)

    return render_template("landing.html", results=[])

@app.route("/query/", methods=["post"])
def query():
    data_text = request.form['query-box']
    data_rows = [row for row in data_text.split("\r\n")]
    data = np.array([[float(n) for n in row.split(",")] for row in data_rows])

    query_results = [learner.query(data) for learner in LEARNERS_LIST]

    return render_template("landing.html", results=query_results)

    return str(query_results)

if __name__ == "__main__":
    app.run(debug=True)