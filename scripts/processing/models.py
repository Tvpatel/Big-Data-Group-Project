import pickle
from sklearn import linear_model
from sklearn.svm import SVR
import csv


class Predictions:

    def __init__(self):
        # self.vector = pickle.load(open('vector.pkl', 'rb').read())
        # self.labels_rating = pickle.load(open('labels_rating.pkl', 'rb').read())
        # self.labels_score = pickle.load(open('labels_score.pkl', 'rb').read())
        # self.train = pickle.load(open('train.pkl', 'rb').read())
        with open(r"labels_rating.pkl", "rb") as input_file:
            self.labels_rating = pickle.load(input_file)
        with open(r"labels_score.pkl", "rb") as input_file:
            self.labels_score = pickle.load(input_file)
        with open(r"train.pkl", "rb") as input_file:
            self.train = pickle.load(input_file)

        # Actually predict
        print "\nPredicting now ....."
        self.predict(self.train, self.labels_score, "score")
        self.predict(self.train, self.labels_rating, "rating")

    def predict(self, train, labels, type):
        # --------------------------------------------------------------
        # Linear Regression
        reg = linear_model.LinearRegression()
        reg.fit(train, labels)

        results = []
        for index in range(0, len(train)):
            result = reg.predict(train[index])[0]
            latitude = train[index][531]
            longitude = train[index][532]
            results.append({'value': result, 'lat': latitude, 'lon': longitude})
        keys = ["value", "lat", "lon"]
        with open('linear_regression_' + type + '.csv', 'wb') as output:
            dict_writer = csv.DictWriter(output, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

        # --------------------------------------------------------------
        # Ridge Regression
        reg = linear_model.Ridge(alpha=0.25)
        reg.fit(train, labels)

        results = []
        for index in range(0, len(train)):
            result = reg.predict(train[index])[0]
            latitude = train[index][531]
            longitude = train[index][532]
            results.append({'value': result, 'lat': latitude, 'lon': longitude})
        keys = ["value", "lat", "lon"]
        with open('ridge_regression_' + type + '.csv', 'wb') as output:
            dict_writer = csv.DictWriter(output, keys)
            dict_writer.writerows(results)

        # --------------------------------------------------------------
        # Lasso Regression
        reg = linear_model.Lasso(alpha=0.1)
        reg.fit(train, labels)

        results = []
        for index in range(0, len(train)):
            result = reg.predict(train[index])[0]
            latitude = train[index][531]
            longitude = train[index][532]
            results.append({'value': result, 'lat': latitude, 'lon': longitude})
        keys = ["value", "lat", "lon"]
        with open('lasso_regression_' + type + '.csv', 'wb') as output:
            dict_writer = csv.DictWriter(output, keys)
            dict_writer.writerows(results)

        # --------------------------------------------------------------
        # Bayesian Ridge Regression
        reg = linear_model.BayesianRidge()
        reg.fit(train, labels)

        results = []
        for index in range(0, len(train)):
            result = reg.predict(train[index])[0]
            latitude = train[index][531]
            longitude = train[index][532]
            results.append({'value': result, 'lat': latitude, 'lon': longitude})
        keys = ["value", "lat", "lon"]
        with open('bayesian_ridge_regression_' + type + '.csv', 'wb') as output:
            dict_writer = csv.DictWriter(output, keys)
            dict_writer.writerows(results)

        # --------------------------------------------------------------
        # Support Vector Regression
        # reg = SVR(kernel='linear', C=1e3)
        # reg.fit(train, labels)

        # results = []
        # for index in range(0, len(train)):
        #     result = reg.predict(train[index])[0]
        #     latitude = train[index][531]
        #     longitude = train[index][532]
        #     results.append({'value': result, 'lat': latitude, 'lon': longitude})
        # keys = ["value", "lat", "lon"]
        # with open('support_vector_regression_' + type + '.csv', 'wb') as output:
        #   dict_writer = csv.DictWriter(output, keys)
        #   dict_writer.writerows(results)
