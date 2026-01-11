import sys
from nnunetv2.run.run_training_nodeepsupervision_mymodel import run_training_entry

if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(run_training_entry())
