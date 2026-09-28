"""Fine-tune MobileNetV2 for ESP32-CAM plant-presence classification."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


SEED = 20260927
CLASS_NAMES = ["not_plant", "plant"]


def load_split(root: Path, split: str, image_size: int, batch_size: int, shuffle: bool):
    dataset = tf.keras.utils.image_dataset_from_directory(
        root / "images" / split,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
        interpolation="bilinear",
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(image_size: int) -> tf.keras.Model:
    inputs = tf.keras.Input((image_size, image_size, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        alpha=0.35,
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="plant_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_035_plant_presence")
    return model


def get_class_weights(dataset_root: Path) -> dict[int, float]:
    counts = {
        index: len(list((dataset_root / "images" / "train" / name).glob("*.jpg")))
        for index, name in enumerate(CLASS_NAMES)
    }
    total = sum(counts.values())
    return {index: total / (2.0 * count) for index, count in counts.items()}


def confusion_metrics(y_true: list[int], probabilities: list[float]) -> dict[str, float | int]:
    y_pred = [int(value >= 0.5) for value in probabilities]
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "true_not_plant": tn,
        "not_plant_predicted_as_plant": fp,
        "plant_predicted_as_not_plant": fn,
        "true_plant": tp,
        "accuracy": (tp + tn) / max(1, len(y_true)),
        "precision_plant": precision,
        "recall_plant": recall,
        "recall_not_plant": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def representative_data(dataset):
    def generator():
        yielded = 0
        for images, _ in dataset:
            for image in images:
                yield [tf.expand_dims(tf.cast(image, tf.float32), 0)]
                yielded += 1
                if yielded >= 100:
                    return
    return generator


def export_tflite(model: tf.keras.Model, train, run_dir: Path) -> dict[str, int | str]:
    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_model = float_converter.convert()
    float_path = run_dir / "model_float32.tflite"
    float_path.write_bytes(float_model)

    int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = representative_data(train)
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_converter.inference_input_type = tf.uint8
    int8_converter.inference_output_type = tf.uint8
    int8_model = int8_converter.convert()
    int8_path = run_dir / "model_int8.tflite"
    int8_path.write_bytes(int8_model)
    return {
        "float32_path": str(float_path),
        "float32_bytes": len(float_model),
        "int8_path": str(int8_path),
        "int8_bytes": len(int8_model),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("tinyml_plant_presence_dataset"))
    parser.add_argument("--run-dir", type=Path, default=Path("training_runs/plant_presence_mobilenetv2_035"))
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "labels.txt").write_text("not_plant\nplant\n", encoding="utf-8")
    train = load_split(args.dataset, "train", args.image_size, args.batch_size, True)
    validation = load_split(args.dataset, "validation", args.image_size, args.batch_size, False)
    test = load_split(args.dataset, "test", args.image_size, args.batch_size, False)

    model = build_model(args.image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    config = {
        "model": "MobileNetV2 alpha 0.35, ImageNet backbone frozen",
        "dataset": str(args.dataset),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "seed": SEED,
        "class_names": CLASS_NAMES,
        "class_weights": get_class_weights(args.dataset),
        "tensorflow_version": tf.__version__,
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        "parameter_count": model.count_params(),
    }
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model.summary(print_fn=print)

    callbacks = [
        tf.keras.callbacks.CSVLogger(args.run_dir / "training_log.csv"),
        tf.keras.callbacks.TensorBoard(log_dir=args.run_dir / "tensorboard", histogram_freq=0),
        tf.keras.callbacks.ModelCheckpoint(
            args.run_dir / "best.keras", monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]
    history = model.fit(
        train,
        validation_data=validation,
        epochs=args.epochs,
        class_weight=config["class_weights"],
        callbacks=callbacks,
        verbose=2,
    )
    model.save(args.run_dir / "final.keras")
    (args.run_dir / "history.json").write_text(
        json.dumps({k: [float(v) for v in values] for k, values in history.history.items()}, indent=2),
        encoding="utf-8",
    )

    keras_test = model.evaluate(test, verbose=0, return_dict=True)
    y_true: list[int] = []
    probabilities: list[float] = []
    for images, labels in test:
        predictions = model.predict(images, verbose=0).reshape(-1)
        y_true.extend(int(value) for value in labels.numpy().reshape(-1))
        probabilities.extend(float(value) for value in predictions)

    metrics: dict[str, object] = {
        "keras_test": {key: float(value) for key, value in keras_test.items()},
        "confusion_at_0.5": confusion_metrics(y_true, probabilities),
        "epochs_completed": len(history.history["loss"]),
    }
    with (args.run_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "true_label", "plant_probability", "predicted_label"],
        )
        writer.writeheader()
        for index, (truth, probability) in enumerate(zip(y_true, probabilities)):
            writer.writerow(
                {
                    "index": index,
                    "true_label": CLASS_NAMES[truth],
                    "plant_probability": f"{probability:.8f}",
                    "predicted_label": CLASS_NAMES[int(probability >= 0.5)],
                }
            )

    metrics["exports"] = export_tflite(model, train, args.run_dir)
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
