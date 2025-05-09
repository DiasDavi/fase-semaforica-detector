import os

BASE_PATH = "dataset"
IMAGES_PATH = [
    os.path.sep.join([BASE_PATH, "dayClip1"]),
    os.path.sep.join([BASE_PATH, "dayClip2"]),
    os.path.sep.join([BASE_PATH, "dayClip3"])
]
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations.json"])

BASE_OUTPUT = "output"

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_files.txt"])

INIT_LR = 1e-4
NUM_EPOCHS = 5
BATCH_SIZE = 32