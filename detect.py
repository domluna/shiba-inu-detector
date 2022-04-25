import requests
import torch
from PIL import Image, ImageDraw
from transformers import DetrConfig, DetrFeatureExtractor, DetrForObjectDetection

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
output_dict = feature_extractor.post_process(
    outputs, torch.tensor([(image.height, image.width)])
)[0]

# config = DetrConfig.from_pretrained('facebook/detr-resnet-50', num_labels=1)


def plot_boxes(image, output_dict, id2label, score_threshold=0.8):
    img = image.copy()
    drawer = ImageDraw.Draw(img)
    labels = output_dict["labels"].tolist()
    scores = output_dict["scores"].tolist()
    bboxes = output_dict["boxes"].tolist()

    for label, score, bbox in zip(labels, scores, bboxes):
        if label not in id2label or score < score_threshold:
            continue

        label = id2label[label]
        print(label, bbox)

        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        print("here")
        drawer.rectangle(((xmin, ymin), (xmax, ymax)), outline="blue")

    img.show()


plot_boxes(image, output_dict, model.config.id2label)

# https://github.com/lucidrains/PaLM-pytorch/blob/main/palm_pytorch/palm_pytorch.py
#
# # https://arxiv.org/abs/2104.09864
#
# class RotaryEmbedding(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
#         self.register_buffer("inv_freq", inv_freq)
#
#     def forward(self, max_seq_len, *, device):
#         seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
#         # returns a tensor of shape (max_seq_len, dim / 2)
#         # the same as doing for i in arr1, j in arr2 i * j
#         freqs = einsum("i , j -> i j", seq, self.inv_freq)
#         # returns a tensor of shape (max_seq_len, dim)
#         return torch.cat((freqs, freqs), dim=-1)
#
#
# def rotate_half(x):
#     x = rearrange(x, "... (j d) -> ... j d", j=2)
#     x1, x2 = x.unbind(dim=-2)
#     return torch.cat((-x2, x1), dim=-1)
#
# def apply_rotary_pos_emb(pos, t):
#     return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# Step 2
#
# Train on plastic/garbage dataset

# datasets consists of images bounding box annotations
# we have two classes/labels 1 for garbage/plastic/trash/etc and 0 for nothing
# basically if garbage is detected in an area of the image, then the label is 1 otherwise 0
# an image may have >= 0 annotations

import json

with open("UAVVaste/annotations/annotations.json") as f:
    data = json.load(f)

# data has keys: "images" & "annotations"

# image sample

# {'id': 771,
#  'width': 3840,
#  'height': 2160,
#  'file_name': 'BATCH_d08_img_1060.jpg',
#  'license': None,
#  'flickr_url': 'https://live.staticflickr.com/65535/50678058848_9e8486b817_o.jpg',
#  'coco_url': None,
#  'date_captured': None,
#  'flickr_640_url': None}]

# we care about "file_name"

# annotation sample

# {'id': 3717,
#  'image_id': 771,
#  'category_id': 0,
#  'segmentation': [[2981, 2100, 3009, 2131, 2967, 2156, 2897, 2159]],
#  'area': 2940.0,
#  'bbox': [2897, 2100, 111, 58],
#  'iscrowd': 0}

# we only care about the "image_id" and "bbox"

annotations = data["annotations"]
images = data["images"]

from collections import defaultdict

d = defaultdict(list)
for a in annotations:
    d[a["image_id"]].append(a["bbox"])

data_list = []
for i in images:
    dd = {}
    dd["image_file"] = i["file_name"]
    dd["bounding_boxes"] = d[i["id"]]
    data_list.append(dd)

with open("uavwaste_data.json", "w") as f:
    json.dump(data_list, f)

with open("uavwaste_data.json", "r") as f:
    data = json.load(f)

import os


class Foo:
    def __init__(self, images_dir, data_file):
        self.data_file = data_file
        self.images_dir = images_dir
