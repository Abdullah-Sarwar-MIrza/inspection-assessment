import argparse
import json
import torch
from .model import Predictor, decode_image
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-dir", default="artifacts")
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(Predictor(args.model_dir).predict(decode_image(args.image.read_bytes())), indent=2))


if __name__ == "__main__":
    main()
