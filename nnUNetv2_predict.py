#!/home/xujialiu/miniconda3/envs/dinov3/bin/python3
# -*- coding: utf-8 -*-
import sys
from nnunetv2.inference.predict_from_raw_data import predict_entry_point
if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(predict_entry_point())