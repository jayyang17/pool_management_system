from ultralytics import YOLO
from src.utils.utils import read_yaml, save_model, get_latest_yolo_run
from src.constants import CONFIG_FILE_PATH
from pathlib import Path

config = read_yaml(CONFIG_FILE_PATH)
s3_model_path = config["S3"]["S3_MODEL_PATH"]

def train_yolo():
    """
    Train a YOLO model using configuration settings and save the best model.
    """
    # Load configuration
    config = read_yaml(CONFIG_FILE_PATH)
    MODEL_CONFIG = config["MODEL"]
    TRAINING_CONFIG = config["TRAINING"]

    # Model settings
    pretrain_model = MODEL_CONFIG["PRETRAIN_MODEL"]
    data_yaml = MODEL_CONFIG["DATA_YAML_PATH"]

    # Training parameters
    img_size = TRAINING_CONFIG["IMG_SIZE"]
    epochs = TRAINING_CONFIG["NUM_EPOCHS"]
    batch_size = TRAINING_CONFIG["BATCH_SIZE"]
    lr0 = TRAINING_CONFIG["LR0"]
    weight_decay = TRAINING_CONFIG["WEIGHT_DECAY"]
    patience = TRAINING_CONFIG["PATIENCE"]

    # Model output paths
    model_output = Path(config["MODEL_OUTPUT_DIR"]).resolve()
    runs_path = Path(config["RUN_PATH"]).resolve()
    model_weights = config["MODEL_WEIGHTS"]

    # Load YOLO model
    model = YOLO(pretrain_model)

    # Start training
    print(f"Starting training with {pretrain_model} on {data_yaml} for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=lr0,
        weight_decay=weight_decay,
        patience=patience
    )

    print("Training complete!")

    # Get latest YOLO run
    latest_yolo_run = get_latest_yolo_run(runs_path)
    if latest_yolo_run:
        save_model(source_train_dir=latest_yolo_run,
                   model_weights=model_weights,
                   output_dir=model_output)

    print(f"Model saved to {model_output}")


if __name__ == "__main__":
    train_yolo()
