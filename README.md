# ContextVIT

This repository is a reorganized version of an original single-file training script. 
The code has been split into logical modules located under `contextvit/` and `modules/`.

`contextvit/` contains data loading, model definitions, configuration utilities and training routines while `modules/` provides lightweight placeholder implementations for the external dependencies referenced in the original notebook.

The main entry point remains `main.py` which simply configures the environment and launches training.
