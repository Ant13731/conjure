import pickle
import tensorflow as tf
from keras import datasets, layers, models, Model, callbacks
import numpy as np
import pickle
import json

import kagglehub
import pprint
from PIL import Image


def load_dataset() -> tuple[str, dict[int, dict[str, np.ndarray]]]:
    path = kagglehub.dataset_download("soumikrakshit/rhp-dataset")
    # print(path)
    training_data_pickle_path = path + "/training/anno_training.pickle"
    training_annotations = pickle.load(open(training_data_pickle_path, "rb"))
    return path, training_annotations
    # pprint.pprint(len(list(training_annotations.keys())))
    # [pprint.pprint(training_annotations[i]) for i in range(10)]
    # dataset annotations: K, uv_vis, xyz,


def preprocess_dataset(
    path: str,
    training_annotations: dict[int, dict[str, np.ndarray]],
    limit: int | None = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    images = []
    uv_datapoints = []
    for i in training_annotations:
        if limit is not None and i >= limit:
            print("Limiting to first", limit, "images")
            break

        if i % 1000 == 0:
            print(f"Processing {i}/{len(list(training_annotations.keys()))}...")
        # uv points are scaled to 320x320, so we normalize them to 0-1 range
        # if training_annotations[i]["uv_vis"].shape[0] != 42:
        #     print(f"Skipping {i} because uv_vis shape is not 42 (is {training_annotations[i]['uv_vis'].shape[0]})")
        #     continue

        # print(training_annotations[i]["uv_vis"].shape)
        # print(training_annotations[i]["uv_vis"])

        image = Image.open(f"{path}/training/color/{i:05}.png")
        formatted_image = np.asarray(image)
        images.append(formatted_image)

        normalized_uv = training_annotations[i]["uv_vis"][:21, :2] / 320.0
        # print(normalized_uv.shape)
        # print(normalized_uv)
        # normalized_uv_with_handedness = np.hstack((normalized_uv, np.array([[0.0]] * 21 + [[1.0]] * 21)))
        # print(normalized_uv_with_handedness.shape)
        # print(normalized_uv_with_handedness)
        # uv_datapoints[i] = normalized_uv_with_handedness
        uv_datapoints.append(normalized_uv)

    return np.array(images), np.array(uv_datapoints)


def get_model() -> Model:
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(320, 320, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(128, activation="relu"))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Dense(21 * 2, activation="relu"))  # 21 points, 2 coordinates (x, y)
    model.add(layers.Reshape((21, 2)))  # Reshape to (21, 2) for the output layer

    model.compile(optimizer="adam", loss="mse")
    return model


callbacks_ = [callbacks.ModelCheckpoint("landmark_model_val_loss.keras", save_best_only=True, monitor="val_loss")]

path, dataset_dict = load_dataset()
dataset_x, dataset_y = preprocess_dataset(path, dataset_dict)
model = get_model()
model.summary()
history = model.fit(
    dataset_x,
    dataset_y,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
)
model.save("landmark_model.keras")
print(history)
print(history.history)
pickle.dump(
    history,
    open("landmark_model_history.pickle", "wb"),
)
json.dump(
    history.history,
    open("landmark_model_history.json", "w"),
)
