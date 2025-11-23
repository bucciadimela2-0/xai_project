import json
import logging
import os
from typing import Dict, List, Optional

import pandas as pd


def save_run_config_sampling(
    output_path: str,
    input_path: str,
    n_samples: int,
    random_state: int,
    command: str,
) -> None:
   
    #Appends the current sampling execution details to run_config_sampling.json
    #inside the output directory. Creates the file if it does not exist.
    

    config_entry = {
        "command": command,
        "input_path": input_path,
        "output_path": output_path,
        "n_samples": n_samples,
        "random_state": random_state,
    }

    run_config_path = os.path.join(os.path.dirname(output_path), "run_config_sampling.json")

    if os.path.exists(run_config_path):
        try:
            with open(run_config_path, "r") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    else:
        existing = []

    existing.append(config_entry)

    with open(run_config_path, "w") as f:
        json.dump(existing, f, indent=4)

    logging.info(f"Sampling run configuration appended to: {run_config_path}")