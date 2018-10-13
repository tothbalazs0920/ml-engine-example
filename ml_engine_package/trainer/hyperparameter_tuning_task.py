import argparse
import os
import tensorflow as tf
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
import keras
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.backend import relu, softmax
import pandas as pd
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.lib.io import file_io


def generate_batches(x_file, y_file, batch_size):
    while True:
        x_chunk = pd.read_csv(tf.gfile.Open(
            x_file), chunksize=batch_size, delimiter="\t")
        y_chunk = pd.read_csv(tf.gfile.Open(
            y_file), chunksize=batch_size, delimiter="\t")
        for x, y in zip(x_chunk, y_chunk):
            yield ({'feature_vector': x.iloc[:x.shape[0]]}, {'predictions': y.iloc[:x.shape[0]]})


def create_model(number_of_features, number_of_nodes_one, number_of_nodes_two):
    """Create a Keras funcional model with layers."""

    feature_vector = Input(shape=(number_of_features,), name='feature_vector')

    layer1 = Dense(number_of_nodes_one, activation='relu')(feature_vector)
    layer2 = Dense(number_of_nodes_two, activation='relu')(layer1)
    predictions = Dense(2, activation='softmax', name='predictions')(layer2)

    model = Model(inputs=[feature_vector], outputs=[predictions])
    model.summary()
    model.compile(optimizer='adam', loss={'predictions':'binary_crossentropy'},
                  metrics=['accuracy'])
    return model


def save_tensorflow_model(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""
    
    if file_io.file_exists(export_path):
        return
    builder = saved_model_builder.SavedModelBuilder(export_path)
    signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                      outputs={'output': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            }
        )
        builder.save()

def save_keras_model(model, export_path, MODEL_FILENAME = 'model.hdf5'):
    if file_io.file_exists(export_path):
        return
    model.save(MODEL_FILENAME)
    tf.gfile.Copy(MODEL_FILENAME,
        export_path +  MODEL_FILENAME)


def run(job_dir,
        x_train_file,
        y_train_file,
        x_test_file,
        y_test_file,
        number_of_epochs,
        number_of_features,
        number_of_training_examples,
        number_of_test_examples,
        batch_size,
        model_location,
        number_of_nodes_one = 10,
        number_of_nodes_two = 10):

    model = create_model(number_of_features, number_of_nodes_one, number_of_nodes_two)

    callbacks = [TensorBoard(job_dir + '/' + 'logs')]

    train_generator = generate_batches(
        x_train_file, y_train_file, batch_size= batch_size)
    
    evaluation_generator = generate_batches(
        x_test_file, y_test_file, batch_size= batch_size)

    model.fit_generator(train_generator,
                    steps_per_epoch= -(-number_of_training_examples // batch_size),
                    epochs= number_of_epochs,
                    validation_data= evaluation_generator,
                    validation_steps= -(-number_of_test_examples // batch_size),
                    callbacks= callbacks)

    evaluation_result = model.evaluate_generator(evaluation_generator, steps= -(-number_of_test_examples // batch_size))
    print('Test loss:', evaluation_result[0])
    print('Test accuracy:', evaluation_result[1])
    
    save_keras_model(model, job_dir + 'keras_models/' +  model_location   + '/')
    save_tensorflow_model(model, job_dir + 'tensorflow_models/' +  model_location   + '/')
    print("ready")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints')
    parser.add_argument('--x_train_file',
                        required=True,
                        type=str,
                        help='path of training samples')
    parser.add_argument('--y_train_file',
                        required=True,
                        type=str,
                        help='path of target result of training samples')
    parser.add_argument('--x_test_file',
                        required=True,
                        type=str,
                        help='path of training samples')
    parser.add_argument('--y_test_file',
                        required=True,
                        type=str,
                        help='path of target result of test samples')
    parser.add_argument('--number_of_epochs',
                        required=True,
                        type=int,
                        help='number of epochs')
    parser.add_argument('--number_of_features',
                        required=True,
                        type=int,
                        help='number of features')
    parser.add_argument('--number_of_training_examples',
                        required=True,
                        type=int,
                        help='number of training examples')
    parser.add_argument('--number_of_test_examples',
                        required=True,
                        type=int,
                        help='number of validation examples')
    parser.add_argument('--batch_size',
                        required=True,
                        type=int,
                        help='batch size')
    parser.add_argument('--model_location',
                        required=True,
                        type=str,
                        help='model location')
    parser.add_argument('--number_of_nodes_one',
                        required=True,
                        type=int,
                        help='number of nodes on the first hidden layer')
    parser.add_argument('--number_of_nodes_two',
                        required=True,
                        type=int,
                        help='number of nodes on the second hidden layer')
                        

    parse_args, unknown = parser.parse_known_args()
    run(**parse_args.__dict__)
