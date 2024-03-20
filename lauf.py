import os
from oss2_uploader import folder_downloader, folder_uploader


"""
    python infer_video_d2.py \
        --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
        --output-dir /home/ecs-user/VideoPose3D/detectron2d \
        --image-ext mp4 \
        /home/ecs-user/VideoPose3D/videos

    python prepare_data_2d_custom.py -i /path/to/detections/output_directory -o myvideos

    python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject input_video.mp4 --viz-action custom --viz-camera 0 --viz-video /path/to/input_video.mp4 --viz-output output.mp4 --viz-size 6

"""
