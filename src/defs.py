import os
import src

SRC_DIR = os.path.dirname(src.__file__)
DATA_DIR = os.path.join(SRC_DIR,"../data")
NET_DIR = os.path.join(SRC_DIR,"../networks")
SAVED_MODELS_DIR = os.path.join(NET_DIR,"saved_models")