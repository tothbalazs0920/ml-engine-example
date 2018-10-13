import os
import json
import numpy as np
from googleapiclient import discovery
from google.oauth2 import service_account
from httplib2 import Http
import pickle
from keras.preprocessing.text import Tokenizer


ML_ENGINE_URL = 'projects/percent-mlengineexample/models/newsgroupclassification/versions/v1'


def label_to_string(predictions):
    result = np.argmax(predictions)
    if result == 0:
        return 'alt.atheism'
    elif result == 1:
        return 'talk.religion.misc'
    elif result == 2:
        return 'comp.graphics'
    return 'sci.space'


def predict(instances):
    """Send json data to a deployed model for prediction."""

    service = discovery.build('ml', 'v1', cache_discovery=False)
    response = service.projects().predict(
        name=ML_ENGINE_URL,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions'][0]['output'], label_to_string(response['predictions'][0]['output'])


def main():
    with open('data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    review = "This is a test newsgroup."
    encoded_docs = tokenizer.texts_to_matrix(review, mode='count')
    values, prediction = predict([{"input": encoded_docs[0].tolist()}])

    print(prediction)
    print(values)


if __name__ == '__main__':
    main()
