import pandas
import pickle
from sklearn.feature_extraction import DictVectorizer

class Provider:

    def get_data(self):
        # Read from CSV
        train_csv = pandas.read_csv("yelp_final.csv", sep=',', header=None)

        for p in range(1, train_csv.shape[0]):
            train_csv[3][p] = int(train_csv[3][p])
            train_csv[4][p] = int(train_csv[4][p])
            train_csv[5][p] = float(train_csv[5][p])
            train_csv[6][p] = float(train_csv[6][p])
            train_csv[7][p] = float(train_csv[7][p])
            train_csv[9][p] = float(train_csv[9][p])
            train_csv[10][p] = int(train_csv[10][p])
            train_csv[8][p] = int(train_csv[8][p].replace('/', ''))
        # Construct list of Dictionaries
        train = []
        for index in range(1, train_csv.shape[0]):
            train.append({train_csv[key][0]: train_csv[key][index] for key in range(0, train_csv.shape[1])})

        labels_rating = []
        labels_score = []
        for index in range(1, train_csv.shape[0]):
            labels_rating.append(float(train_csv[5][index]))
            labels_score.append(float(train_csv[4][index]))

        # Construct sparse vector
        vector_maker = DictVectorizer()
        train_vector = vector_maker.fit_transform(train).toarray()

        # For reference print the labels for vector
        vector_maker.get_feature_names()

        # Save Data into files
        with open('train.pkl', 'wb') as output:
            pickle.dump(train_vector, output, pickle.HIGHEST_PROTOCOL)
        print "\nTraining data saved"

        with open('labels_rating.pkl', 'wb') as output:
            pickle.dump(labels_rating, output, pickle.HIGHEST_PROTOCOL)
        print "\nRating labels saved"

        with open('labels_score.pkl', 'wb') as output:
            pickle.dump(labels_score, output, pickle.HIGHEST_PROTOCOL)
        print "\nScore labels saved"

        with open('vector.pkl', 'wb') as output:
            pickle.dump(vector_maker, output, pickle.HIGHEST_PROTOCOL)
        print "\nVector saved"

        # Return result
        return train_vector, labels_rating, labels_score, vector_maker
