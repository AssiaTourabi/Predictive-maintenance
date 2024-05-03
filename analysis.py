from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        AirTemperature = float(request.form['AirTemperature'])
        ProcessTemperature = float(request.form['ProcessTemperature'])
        RotationalSpeed = float(request.form['RotationalSpeed'])
        Torque = float(request.form['Torque'])
        ToolWear = float(request.form['ToolWear'])
        TWF = float(request.form['TWF'])
        HDF = float(request.form['HDF'])
        PWF = float(request.form['PWF'])
        OSF = float(request.form['OSF'])
        RNF = float(request.form['RNF'])

        # Charger le dataset
        data = pd.read_csv(r"data.csv")
       

        features = ['Air temperature [K]', 'Process temperature [K]',
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                    'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

        X = data[features]
        y = data['Machine failure']

        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialiser et entraîner le modèle de régression logistique
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Faire des prédictions sur l'ensemble de test
        predictions = model.predict(X_test)

        # Évaluer les performances du modèle
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy}')

        # Afficher le rapport de classification
        print(classification_report(y_test, predictions))

        new_values = np.array([[AirTemperature, ProcessTemperature, RotationalSpeed, Torque, ToolWear, TWF, HDF, PWF, OSF, RNF]])

        # Faire une prédiction avec le modèle entraîné
        prediction = model.predict(new_values)

        result = ""
        if prediction == 0:
            result = "Negatif"
        elif prediction == 1:
            result = "Positive"
        return render_template('index.html', result=result)
    else:
        result = ""
        return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
