from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import pandas as pd
import pickle
import os

app = Flask(__name__)
CORS(app)

# Carrega o modelo
with open(os.path.join('model', 'NAIVE_BAYES.pkcls'), 'rb') as file:
    model = pickle.load(file)

# Página inicial
@app.route('/')
def index():
    return render_template('index.html')

# Página do formulário
@app.route('/formulario')
def formulario():
    return render_template('form.html')

# Rota de predição
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    data_sex = data['sex']  # 0 feminino, 1 masculino

    input_data = pd.DataFrame([{
        'age': data['age'],
        'sex_0': 1 if data_sex == 0 else 0,
        'sex_1': 1 if data_sex == 1 else 0,
        'cp': data['cp'],
        'chol': data['chol'],
        'trtbps': data['trtbps'],
        'thalachh': data['thalachh'],
    }])

    prediction = model.predict(input_data)[0]
    return jsonify({"result": int(prediction)})

# Rota para métricas
@app.route('/metrics', methods=['GET'])
def metrics():
    # Carrega os dados de teste
    test_data = pd.read_csv(os.path.join('data', 'heart.csv'))

    # Criação de variáveis dummy para o sexo
    test_data['sex_0'] = test_data['sex'].apply(lambda x: 1 if x == 0 else 0)
    test_data['sex_1'] = test_data['sex'].apply(lambda x: 1 if x == 1 else 0)

    # Seleção de features
    X_test = test_data[['age', 'sex_0', 'sex_1', 'cp', 'chol', 'trtbps', 'thalachh']].values
    y_test = test_data['target']

    # Predição
    y_pred = model.predict(X_test)
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
     
    print("X_test shape:", X_test.shape)
    print("y_pred shape:", y_pred.shape)


    # Cálculo das métricas
    acc = accuracy_score(y_test, y_pred)
    sens = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    espec = tn / (tn + fp)

    return jsonify({
        "accuracy": round(acc, 2),
        "sensitivity": round(sens, 2),
        "specificity": round(espec, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
