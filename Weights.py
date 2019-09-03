import numpy as np
import pickle

face_1 = np.load('face_1.npy')
face_2 = np.load('face_2.npy')
face_1 = face_1.reshape(400,-1)
face_2 = face_2.reshape(400,-1)
faces = np.concatenate([kunal, samyak])
faces = faces/255.
labels = np.zeros((faces.shape[0], 1))
labels[400:, :] = 1.0
print(labels)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(x, w):
    z = np.dot(x, w)
    return sigmoid(z)


def gradientDescent(x_train, y_train, epochs, alpha):
    w = np.zeros(x_train.shape[1])
    n = len(x_train)
    for epoch in range(epochs):
        pred = predict(x_train, w)
        loss = pred - y_train.flatten()
        grad = (1 / n) * x_train.T.dot(loss)
        w = w - alpha * grad
    return w


def logistic(x_train, y_train, epochs, alpha):
    w = gradientDescent(x_train, y_train, epochs, alpha)
    file = open('weights.pkl', 'wb')
    pickle.dump(w, file)
    prediction = predict(x_train, w)
    for i in range(len(prediction)):
        prediction[i] = round(prediction[i])
    return prediction


def accuracy(predictions, actual):
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == actual[i]:
            count += 1
    return count / len(predictions) * 100


def evaluate(epochs, alpha):
    pred = logistic(faces, labels, epochs, alpha)
    s = accuracy(pred, labels)
    return s


epochs = 10000
alpha = 0.03
score = evaluate(epochs, alpha)
print(score)
