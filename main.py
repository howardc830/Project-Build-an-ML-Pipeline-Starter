import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config', config_path=".", version_base="1.3")
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    #raw_artifact_name = f"{config['etl']['sample']}:latest"
    #if any(step in active_steps for step in ["basic_cleaning", "data_check", "data_split", "train_random_forest", "test_regression_model"]):
       # try:
        #    wandb.Api().artifacts(f"{os.environ['WANDB_PROJECT']}/{raw_artifact_name}")
        #except wandb.errors.CommError:
         #   print(f"raw data {raw_artifact_name} not found. Running 'download' step automatically.")
          #  active_steps = ["download"] + [s for s in active_steps if s != "download"]

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
               # uri="./components/get_data",
                f"{config['main']['components_repository']}/get_data",
                'main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            print("Running basic cleaning step...")
            _ = mlflow.run(
                "./src/basic_cleaning",
                "main",
                env_manager="conda",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data cleaned by removing outliers and invalid entries",
                    "min_price": 10,
                    "max_price": 350,
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                "./src/data_check",
                "main",
                env_manager="conda",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": 0.2,
                    "min_price": 10,
                    "max_price": 350,
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                parameters = {
                    "input": "clean_sample.csv:latest",
                    "test_size": 0.2,
                    "random_seed": 42,
                    "stratify_by": "neighbourhood_group",
                },
            )
           

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step
            _ = mlflow.run(
                "./src/train_random_forest",
                "main",
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": 0.2,
                    "random_seed": 42,
                    "stratify_by": "neighbourhood_group",
                    "rf_config": rf_config,
                    "max_tfidf_features": 5,
                    "output_artifact": "random_forest_export",

                },
            )

        if "test_regression_model" in active_steps:

            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                "main",
                env_manager="conda",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest",

                },
            )


if __name__ == "__main__":
    go()
