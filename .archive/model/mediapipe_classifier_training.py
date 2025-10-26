import tensorflow as tf
import matplotlib.pyplot as plt

# NOTE: Library `mediapipe_model_maker` only works on Linux with python 3.11. It will not run on Windows.
# The dataset is from huggingface.co/cj-mills/hagrid-sample-30k-384p (only the `images` folder, excluding `call, dislike, four, like, mute, three, three2` subfolders)
DATASET_PATH = "/home/ant13731-wsl/3ml3_datasets/hagrid-sample-30k-384p"


# Source: https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer#get_the_dataset
def fine_tune_model():
    from mediapipe_model_maker import gesture_recognizer

    # Fine-tune model
    data = gesture_recognizer.Dataset.from_folder(
        dirname=f"{DATASET_PATH}/images",
        hparams=gesture_recognizer.HandDataPreprocessingParams(),
    )
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    hparams = gesture_recognizer.HParams(
        learning_rate=0.005,
        batch_size=32,
        epochs=50,
        export_dir="trained_mediapipe_gesture_recognizer",
    )
    model_options = gesture_recognizer.ModelOptions(
        dropout_rate=0.1,
        layer_widths=[64, 32],
    )
    options = gesture_recognizer.GestureRecognizerOptions(
        model_options=model_options,
        hparams=hparams,
    )
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options,
    )

    loss, acc = model.evaluate(test_data, batch_size=1)
    print(f"Test loss:{loss}, Test accuracy:{acc}")

    model.export_model()


def read_results(path: str) -> tuple[list[float], list[float]]:
    """Read training/validation loss/accuracies from the tfevent files."""

    accuracy = []
    loss = []
    for summary in tf.compat.v1.train.summary_iterator(path):
        # print(summary)
        for value in summary.summary.value:
            # print(value)
            if value.tag == "epoch_categorical_accuracy":
                accuracy.append(float(tf.io.decode_raw(value.tensor.tensor_content, value.tensor.dtype)[0]))
            elif value.tag == "epoch_loss":
                loss.append(float(tf.io.decode_raw(value.tensor.tensor_content, value.tensor.dtype)[0]))

    return accuracy, loss


def create_graphs() -> None:
    files = [
        r"C:\Users\hunta\Documents\GitHub\conjure\model\trained_mediapipe_gesture_recongizer_data\logs\train\events.out.tfevents.1744947574.Ant13731.163872.0.v2",
        # r"C:\Users\hunta\Documents\GitHub\conjure\model\trained_mediapipe_gesture_recongizer_data\logs\train\events.out.tfevents.1744948269.Ant13731.172518.0.v2", # older files
        r"C:\Users\hunta\Documents\GitHub\conjure\model\trained_mediapipe_gesture_recongizer_data\logs\validation\events.out.tfevents.1744947578.Ant13731.163872.1.v2",
        # r"C:\Users\hunta\Documents\GitHub\conjure\model\trained_mediapipe_gesture_recongizer_data\logs\validation\events.out.tfevents.1744948273.Ant13731.172518.1.v2",
    ]

    accuracyT, lossT = read_results(files[0])
    accuracyV, lossV = read_results(files[1])

    plt.plot(lossT, label="Training Loss")
    plt.plot(lossV, label="Validation Loss")
    plt.title("Training/Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    plt.plot(accuracyT, label="Training Accuracy")
    plt.plot(accuracyV, label="Validation Accuracy")
    plt.title("Training/Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    create_graphs()
    # fine_tune_model()
