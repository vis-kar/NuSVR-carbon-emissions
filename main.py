from flask import Flask, Response, request, json, render_template
import pickle
import numpy as np

app = Flask(__name__)
with open('NuSVR.pkl', 'rb') as f:
  model = pickle.load(f)


@app.route('/', methods=['GET'])
def home():
  return render_template('docs.html')


@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    if request.is_json:
      data = request.json
      transport = data.get('transport')
      electricity = data.get('electricity')
      meals = data.get('meals')
      waste = data.get('waste')
      if not transport or not electricity or not meals or not waste:
        return Response(json.dumps({
            'message':
            'Missing required fields: transport, electricity, meals, waste'
        }),
                        status=400,
                        mimetype='application/json')
      data = [[transport, electricity, meals, waste]]
      data = np.array(data)
      result = model.predict(data)
      return Response(json.dumps({'result': result[0]}),
                      status=200,
                      mimetype='application/json')
    return Response(json.dumps({'message': 'Invalid request'}),
                    status=403,
                    mimetype='appication/json')


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
