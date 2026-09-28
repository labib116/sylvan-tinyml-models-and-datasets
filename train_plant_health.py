"""Train CPU-friendly binary plant-health models and export TFLite artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


SEED = 20260927
CLASS_NAMES = ["healthy", "unhealthy"]


def load_split(root: Path, split: str, image_size: int, batch_size: int, shuffle: bool) -> tf.data.Dataset:
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


def tiny_cnn(image_size: int) -> tf.keras.Model:
    inputs = tf.keras.Input((image_size, image_size, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)
    for filters in (24, 32, 48):
        x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(max_value=6.0)(x)
        x = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(max_value=6.0)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="health_probability")(x)
    return tf.keras.Model(inputs, outputs, name="tiny_plant_cnn")


def mobilenet_v2(image_size: int) -> tf.keras.Model:
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
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="health_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_035_plant_health")
    model.backbone = backbone  # type: ignore[attr-defined]
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def class_weights(dataset_root: Path) -> dict[int, float]:
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
    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "accuracy": accuracy,
        "precision_unhealthy": precision,
        "recall_unhealthy": recall,
        "specificity_healthy": specificity,
        "f1_unhealthy": f1,
    }


def representative_data(dataset: tf.data.Dataset):
    def generator():
        yielded = 0
        for images, _ in dataset:
            for image in images:
                yield [tf.expand_dims(tf.cast(image, tf.float32), 0)]
                yielded += 1
                if yielded >= 100:
                    return
    return generator


def export_tflite(model: tf.keras.Model, train: tf.data.Dataset, run_dir: Path) -> dict[str, int | str]:
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
    parser.add_argument("--model", choices=("tiny_cnn", "mobilenetv2_035"), default="tiny_cnn")
    parser.add_argument("--dataset", type=Path, default=Path("tinyml_contextual_plant_dataset"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--fine-tune-layers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    run_dir = args.run_dir or Path("training_runs") / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "labels.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    train = load_split(args.dataset, "train", args.image_size, args.batch_size, True)
    validation = load_split(args.dataset, "validation", args.image_size, args.batch_size, False)
    test = load_split(args.dataset, "test", args.image_size, args.batch_size, False)

    if args.initial_model:
        model = tf.keras.models.load_model(args.initial_model)
        learning_rate = args.learning_rate or 1e-5
        if args.fine_tune_layers:
            backbone = model.get_layer("mobilenetv2_0.35_96")
            backbone.trainable = True
            for layer in backbone.layers:
                layer.trainable = False
            for layer in backbone.layers[-args.fine_tune_layers:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
    elif args.model == "tiny_cnn":
        model = tiny_cnn(args.image_size)
        learning_rate = args.learning_rate or 1e-3
    else:
        model = mobilenet_v2(args.image_size)
        learning_rate = args.learning_rate or 5e-4
    compile_model(model, learning_rate)

    config = {
        "model": args.model,
        "dataset": str(args.dataset),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "seed": SEED,
        "class_names": CLASS_NAMES,
        "class_weights": class_weights(args.dataset),
        "tensorflow_version": tf.__version__,
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        "parameter_count": model.count_params(),
        "trainable_parameter_count": sum(int(np.prod(weight.shape)) for weight in model.trainable_weights),
        "initial_model": str(args.initial_model) if args.initial_model else None,
        "fine_tune_layers": args.fine_tune_layers,
        "learning_rate": learning_rate,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model.summary(print_fn=lambda line: print(line))

    callbacks = [
        tf.keras.callbacks.CSVLogger(run_dir / "training_log.csv"),
        tf.keras.callbacks.TensorBoard(log_dir=run_dir / "tensorboard", histogram_freq=0),
        tf.keras.callbacks.ModelCheckpoint(
            run_dir / "best.keras", monitor="val_loss", save_best_only=True, verbose=1
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
    model.save(run_dir / "final.keras")
    (run_dir / "history.json").write_text(
        json.dumps({key: [float(v) for v in values] for key, values in history.history.items()}, indent=2),
        encoding="utf-8",
    )

    test_values = model.evaluate(test, verbose=0, return_dict=True)
    y_true: list[int] = []
    probabilities: list[float] = []
    for images, labels in test:
        predictions = model.predict(images, verbose=0).reshape(-1)
        y_true.extend(int(value) for value in labels.numpy().reshape(-1))
        probabilities.extend(float(value) for value in predictions)
    metrics = {"keras_test": {key: float(value) for key, value in test_values.items()}}
    metrics["confusion"] = confusion_metrics(y_true, probabilities)
    metrics["epochs_completed"] = len(history.history["loss"])

    with (run_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "true_label", "unhealthy_probability", "predicted_label"])
        writer.writeheader()
        for index, (truth, probability) in enumerate(zip(y_true, probabilities)):
            writer.writerow(
                {
                    "index": index,
                    "true_label": CLASS_NAMES[truth],
                    "unhealthy_probability": f"{probability:.8f}",
                    "predicted_label": CLASS_NAMES[int(probability >= 0.5)],
                }
            )

    metrics["exports"] = export_tflite(model, train, run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
