
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

DATA_DIR = Path(os.getenv("DATA_DIR"))

INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
VECTOR_INPUT_DIR = INPUT_DIR / "vector"
VECTOR_OUTPUT_DIR = OUTPUT_DIR / "vector"
RASTER_INPUT_DIR = INPUT_DIR / "raster"
RASTER_OUTPUT_DIR = OUTPUT_DIR / "raster"
TABULAR_INPUT_DIR = INPUT_DIR / "tabular"
TABULAR_OUTPUT_DIR = OUTPUT_DIR / "tabular"
SERIALISED_OUTPUT_DIR = OUTPUT_DIR / "serialised"