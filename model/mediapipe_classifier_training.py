from huggingface_hub import HfApi
from mediapipe_model_maker import gesture_recognizer

# NOTE: Library `mediapipe_model_maker` only works on Linux with python 3.11. It will not run on Windows.
# The dataset is from huggingface.co/cj-mills/hagrid-sample-30k-384p (only the `images` folder, excluding `call, dislike, four, like, mute, three, three2` subfolders)

DATASET_PATH = "/home/ant13731-wsl/3ml3_datasets/hagrid-sample-30k-384p"


# Source: https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer#get_the_dataset
def fine_tune_model():
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


if __name__ == "__main__":
    fine_tune_model()
