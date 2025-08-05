import pickle

with open('/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/pickles/preprocess_X_train.pickle', 'rb') as pf:
    X_train = pickle.load(pf)
with open('/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/pickles/preprocess_X_test.pickle',  'rb') as pf:
    X_test = pickle.load(pf)
with open('/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/pickles/preprocess_y_train.pickle', 'rb') as pf:
    y_train = pickle.load(pf)
with open('/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/pickles/preprocess_y_test.pickle',  'rb') as pf:
    y_test = pickle.load(pf)
from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
model = model.fit(X_train,y_train)
with open('/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/pickles/train_model_nn_mlp_c.pickle', 'wb') as pickle_file:
    pickle.dump(model, pickle_file)

y_pred = model.predict(X_test)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

plots_path = '/home/uwubuntu/docker-full-stack/ml-quadrat-backend/src/main/resources/static/storage/test/generated-Cargo/python_java/src/python-scripts/plots/'

plt.figure()
plt.plot(y_train, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Prediction vs Actual')
plt.legend()
plt.savefig(plots_path + 'prediction_vs_actual.png')
plt.close()

