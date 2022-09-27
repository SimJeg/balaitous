import shutil
from train import do_learning

DATA_DIR = "/input/"
ARTIFACT_DIR = "/output/"

if __name__ == "__main__":
    artifacts = do_learning(DATA_DIR)
    for artifact in artifacts:
        shutil.copy(artifact, ARTIFACT_DIR)
