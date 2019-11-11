from emotion_predictor import EmotionPredictor
from flask import Flask, render_template, request
from flask_restful import Resource, Api, reqparse
import pandas as pd
import json


app = Flask(__name__)
app.config['EXPLAIN_TEMPLATE_LOADING'] = True
app.config['DEBUG'] = True

api = Api(app)

# Pandas presentation options
pd.options.display.max_colwidth = 150   # show whole tweet's content
pd.options.display.width = 200          # don't break columns
# pd.options.display.max_columns = 7      # maximal number of columns

# http://www.vstechnologies.net/wp-content/uploads/2018/12/IEEEJV_82Emotion-Recognition-on-Twitter-Comparative-Study-and-Training-a-Unison-Model.pdf
# Predictor for Ekman's emotions in multiclass setting.
# classifications: ekman, plutchik, poms
# setting: mc (multiclass) ml (multilabel)
model = EmotionPredictor(classification='plutchik', setting='mc', use_unison_model=True)

#probabilities = model.predict_probabilities(tweets)
#print(probabilities, '\n')
#embeddings = model.embed(tweets)
#print(embeddings, '\n')

def format_predictions(predictions):
    s=predictions.iloc[0].iloc[1:].sort_values(ascending=False)
    return list(zip(s, s.index))

@app.route('/', methods=["GET", "POST"])
def index_page():
    context = {'one': 1, 'two': 2}
    if request.method == "POST":
        essay =  request.form.get("essay")
        predictions = model.predict_classes([essay])
        probabilities = format_predictions(model.predict_probabilities([essay]))

        context.update({
            'essay':essay,
            'predictions': predictions,
            'probabilities': probabilities,
        })

    return render_template("index.html", **context)

parser = reqparse.RequestParser()
parser.add_argument('essay', help='Returns probabilities for list of emotions according to plutchik model')


class EmotionPredictorResource(Resource):
    def post(self):
        args = parser.parse_args()
        essay = args['essay']

        probabilities = model.predict_probabilities([essay])

        response = {}
        response.update({
            'probabilities': json.loads(probabilities.iloc[0].to_json())})

        return response

api.add_resource(EmotionPredictorResource, '/api/predictor')

if __name__ == '__main__':
    app.run()

