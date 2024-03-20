# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Perform inference on a single video or all videos with a certain extension
(e.g., .mp4) in a folder.
"""

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import subprocess as sp
import numpy as np
import time
import argparse
import sys
import os
import glob
from dotenv import load_dotenv
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

load_dotenv()  # take environment variables from .env.


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end inference")
    parser.add_argument(
        "--cfg",
        dest="cfg",
        help="cfg model file (/path/to/model_config.yaml)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="directory for visualization pdfs (default: /tmp/infer_simple)",
        default="/tmp/infer_simple",
        type=str,
    )
    parser.add_argument(
        "--image-ext",
        dest="image_ext",
        help="image file name extension (default: mp4)",
        default="mp4",
        type=str,
    )
    parser.add_argument(
        "--chunk-num",
        dest="chunk_num",
        help="select a chunk of videos to process (default: 0)",
        default=0,
        type=int,
    )
    parser.add_argument("im_or_folder", help="image or folder of images", default=None)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_resolution(filename):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        filename,
    ]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    for line in pipe.stdout:
        w, h = line.decode().strip().split(",")
        return int(w), int(h)


def read_video(filename):
    w, h = get_resolution(filename)

    command = [
        "ffmpeg",
        "-i",
        filename,
        "-f",
        "image2pipe",
        "-pix_fmt",
        "bgr24",
        "-vsync",
        "0",
        "-vcodec",
        "rawvideo",
        "-",
    ]

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    while True:
        data = pipe.stdout.read(w * h * 3)
        if not data:
            break
        yield np.frombuffer(data, dtype="uint8").reshape((h, w, 3))


def split_array(array, n):
    """Splits an array into n pieces as evenly as possible.

    Args:
        array: The array to split.
        n: The number of pieces to split the array into.

    Returns:
        A list of lists, where each sublist is a piece of the original array.
    """
    chunk_size = len(array) // n  # Integer division for even floor
    remainder = len(array) % n

    pieces = []
    start = 0
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        pieces.append(array[start:end])
        start = end

    return pieces


def get_bucket():
    bucket_name = ("pose-daten",)
    oss_endpoint = ("oss-ap-southeast-1.aliyuncs.com",)

    # 使用环境变量中获取的RAM用户的访问密钥配置访问凭证。
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

    # yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
    # 填写Bucket名称，并设置连接超时时间为30秒。
    bucket = oss2.Bucket(auth, oss_endpoint, bucket_name, connect_timeout=30)

    return bucket


def main(args):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)
    predictor = DefaultPredictor(cfg)

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + "/*." + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    # check local generated results and skip if already exists
    output_generated = set([f.replace(".npz", "") for f in os.listdir(args.output_dir)])

    # check oss and skip if already exists
    bucket = get_bucket()

    oss_prefix = "detectron2d/"

    osskeys = [
        obj.key.replace(".npz", "")
        for obj in oss2.ObjectIterator(bucket, prefix=oss_prefix)
    ]

    remin_video_list = []

    for video_name in im_list:
        # if the basename of the video is already in the output directory, skip
        if os.path.basename(video_name) in output_generated:
            print("{} already generated, skip".format(video_name))
            continue

        if os.path.basename(video_name) in osskeys:
            print(
                f"{oss_prefix}{os.path.basename(video_name)}.npz already exists in oss, skipping."
            )
            continue

        # if not exists, add to remin_video_list
        remin_video_list.append(video_name)

    # split `im_list` into 4 chunks
    remin_video_list = split_array(remin_video_list, 4)
    # get the chunk specified by `args.chunk_num`
    remin_video_list = remin_video_list[args.chunk_num]

    for i, video_name in enumerate(remin_video_list):
        out_name = os.path.join(args.output_dir, os.path.basename(video_name))
        # check if out_name exists
        if os.path.exists(f"{out_name}.npz"):
            print("{} already exists, skip".format(out_name))
            continue

        # check if the file already exists in oss
        if bucket.object_exists(f"{oss_prefix}{out_name}.npz"):
            print(
                f"{oss_prefix}{out_name}.npz already exists in oss, skipping. {i}/{len(remin_video_list)}"
            )
            continue

        # check if file size 0
        if os.stat(video_name).st_size == 0:
            print("{} is empty, skip".format(video_name))
            continue

        if video_name.startswith("videos"):
            print("illegal name, skip")
            continue

        print("Processing {} {}".format(video_name, i))

        boxes = []
        segments = []
        keypoints = []

        for frame_i, im in enumerate(read_video(video_name)):
            t = time.time()
            outputs = predictor(im)["instances"].to("cpu")

            print("Frame {} processed in {:.3f}s".format(frame_i, time.time() - t))

            has_bbox = False
            if outputs.has("pred_boxes"):
                bbox_tensor = outputs.pred_boxes.tensor.numpy()
                if len(bbox_tensor) > 0:
                    has_bbox = True
                    scores = outputs.scores.numpy()[:, None]
                    bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)
            if has_bbox:
                kps = outputs.pred_keypoints.numpy()
                kps_xy = kps[:, :, :2]
                kps_prob = kps[:, :, 2:3]
                kps_logit = np.zeros_like(kps_prob)  # Dummy
                kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
                kps = kps.transpose(0, 2, 1)
            else:
                kps = []
                bbox_tensor = []

            # Mimic Detectron1 format
            cls_boxes = [[], bbox_tensor]
            cls_keyps = [[], kps]

            boxes.append(cls_boxes)
            segments.append(None)
            keypoints.append(cls_keyps)

        # Video resolution
        metadata = {
            "w": im.shape[1],
            "h": im.shape[0],
        }

        # avoid ValueError: setting an array element with a sequence.
        # The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (5, 2) + inhomogeneous part.
        boxes = np.array(boxes, dtype=object)
        keypoints = np.array(keypoints, dtype=object)

        np.savez_compressed(
            out_name,
            boxes=boxes,
            segments=segments,
            keypoints=keypoints,
            metadata=metadata,
        )


if __name__ == "__main__":
    setup_logger()
    args = parse_args()
    main(args)
