# from datasets import load_dataset

# dataset = load_dataset("SwayStar123/CelebV-HQ")

# print(dataset["train"][0])

import os
import cv2
import json
from tqdm import tqdm

def get_size(path):
    v = cv2.VideoCapture(path)
    return f"{int(v.get(cv2.CAP_PROP_FRAME_WIDTH))}_{int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))}"


celeb_meta = json.load(open("datasets/celebvhq_info.json"))
meta_info = celeb_meta["meta_info"]
clips = celeb_meta["clips"]

sizes = set()
for clip in tqdm(clips.keys()):
    sizes.add(get_size(os.path.join("datasets/celebs", clip+".mp4")))

print(sizes)
print(len(sizes))