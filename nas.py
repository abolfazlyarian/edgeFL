import autokeras as ak
from tensorflow.keras.datasets import mnist
import tf2onnx
import onnx
from onnx2pytorch import ConvertModel

def convert_model():

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize the input data
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # Perform Neural Architecture Search
    clf = ak.ImageClassifier(max_trials=1)  # Set the maximum number of trials

    # Search for the optimal architecture and fit the model
    clf.fit(x_train, y_train, epochs=1)

    # Evaluate the best model on the test set
    accuracy = clf.evaluate(x_test, y_test)[1]

    print(f"Accuracy: {accuracy}")

    tf_model = clf.export_model() 
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model)

    # Save the ONNX model
    with open("model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    onnx_model = onnx.load('model.onnx')
    pytorch_model = ConvertModel(onnx_model)

    return pytorch_model