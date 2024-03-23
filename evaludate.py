import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random
from common.custom_dataset import CustomDataset

args = parse_args()
print(args)


print("Loading dataset...")


dataset = CustomDataset("custom_dataset/20240322-2086.npz")

# print(dataset)

print("Loading 2D detections...")
keypoints = np.load("custom_dataset/20240322-2086.npz", allow_pickle=True)
keypoints_metadata = keypoints["metadata"].item()
keypoints_symmetry = keypoints_metadata["keypoints_symmetry"]
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(
    dataset.skeleton().joints_right()
)
keypoints = keypoints["positions_2d"].item()

# print(keypoints)


for subject in dataset.subjects():
    assert (
        subject in keypoints
    ), "Subject {} is missing from the 2D detections dataset".format(subject)
    for action in dataset[subject].keys():
        assert (
            action in keypoints[subject]
        ), "Action {} of subject {} is missing from the 2D detections dataset".format(
            action, subject
        )
        if "positions_3d" not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]["positions_3d"][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][
                    cam_idx
                ][:mocap_length]

        assert len(keypoints[subject][action]) == len(
            dataset[subject][action]["positions_3d"]
        )

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(
                kps[..., :2], w=cam["res_w"], h=cam["res_h"]
            )
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(",")
subjects_semi = (
    [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(",")
)
if not args.render:
    subjects_test = args.subjects_test.split(",")
else:
    subjects_test = [args.viz_subject]

semi_supervised = len(subjects_semi) > 0
if semi_supervised and not dataset.supports_semi_supervised():
    raise RuntimeError("Semi-supervised training is not implemented for this dataset")


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), "Camera count mismatch"
                for cam in cams:
                    if "intrinsic" in cam:
                        out_camera_params.append(cam["intrinsic"])

            if parse_3d_poses and "positions_3d" in dataset[subject][action]:
                poses_3d = dataset[subject][action]["positions_3d"]
                assert len(poses_3d) == len(poses_2d), "Camera count mismatch"
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(
                0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i]))
            )
            out_poses_2d[i] = out_poses_2d[i][start : start + n_frames : stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start : start + n_frames : stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


action_filter = None


cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

filter_widths = [int(x) for x in args.architecture.split(",")]
if not args.disable_optimizations and not args.dense and args.stride == 1:
    # Use optimized model for single-frame predictions
    model_pos_train = TemporalModelOptimized1f(
        poses_valid_2d[0].shape[-2],
        poses_valid_2d[0].shape[-1],
        dataset.skeleton().num_joints(),
        filter_widths=filter_widths,
        causal=args.causal,
        dropout=args.dropout,
        channels=args.channels,
    )
else:
    # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
    model_pos_train = TemporalModel(
        poses_valid_2d[0].shape[-2],
        poses_valid_2d[0].shape[-1],
        dataset.skeleton().num_joints(),
        filter_widths=filter_widths,
        causal=args.causal,
        dropout=args.dropout,
        channels=args.channels,
        dense=args.dense,
    )

model_pos = TemporalModel(
    poses_valid_2d[0].shape[-2],
    poses_valid_2d[0].shape[-1],
    dataset.skeleton().num_joints(),
    filter_widths=filter_widths,
    causal=args.causal,
    dropout=args.dropout,
    channels=args.channels,
    dense=args.dense,
)
