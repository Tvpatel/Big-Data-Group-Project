from provider import Provider
import pickle

train, labels_rating, labels_score, vector = Provider().get_train()
print train[0]

with open('train.pkl', 'wb') as output:
    pickle.dump(train, output, pickle.HIGHEST_PROTOCOL)

with open('labels_rating.pkl', 'wb') as output:
    pickle.dump(labels_rating, output, pickle.HIGHEST_PROTOCOL)

with open('labels_score.pkl', 'wb') as output:
    pickle.dump(labels_score, output, pickle.HIGHEST_PROTOCOL)

with open('vector.pkl', 'wb') as output:
    pickle.dump(vector, output, pickle.HIGHEST_PROTOCOL)