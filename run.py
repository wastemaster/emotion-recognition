from flask import Flask, render_template, request
import pandas as pd
from emotion_predictor import EmotionPredictor

app = Flask(__name__)
app.config['EXPLAIN_TEMPLATE_LOADING'] = True
app.config['DEBUG'] = True

# Pandas presentation options
pd.options.display.max_colwidth = 150   # show whole tweet's content
pd.options.display.width = 200          # don't break columns
# pd.options.display.max_columns = 7      # maximal number of columns

# http://www.vstechnologies.net/wp-content/uploads/2018/12/IEEEJV_82Emotion-Recognition-on-Twitter-Comparative-Study-and-Training-a-Unison-Model.pdf
# Predictor for Ekman's emotions in multiclass setting.
# classifications: ekman, plutchik, poms
# setting: mc (multiclass) ml (multilabel)
model1 = EmotionPredictor(classification='ekman', setting='mc', use_unison_model=True)
model2 = EmotionPredictor(classification='plutchik', setting='mc', use_unison_model=True)
model3 = EmotionPredictor(classification='poms', setting='mc', use_unison_model=True)

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
        predictions1 = model1.predict_classes([essay])
        predictions2 = model2.predict_classes([essay])
        predictions3 = model3.predict_classes([essay])

        probabilities1 = format_predictions(model1.predict_probabilities([essay]))
        probabilities2 = format_predictions(model2.predict_probabilities([essay]))
        probabilities3 = format_predictions(model3.predict_probabilities([essay]))

        context.update({
            'essay':essay,
            'predictions1': predictions1,
            'predictions2': predictions2,
            'predictions3': predictions3,
            'probabilities1': probabilities1,
            'probabilities2': probabilities2,
            'probabilities3': probabilities3
        })

    return render_template("index.html", **context)

if __name__ == '__main__':
    app.run()

