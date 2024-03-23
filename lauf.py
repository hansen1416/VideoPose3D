import os
import glob
import subprocess

from oss2_uploader import folder_downloader, folder_uploader

videos_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "videos")

# iterat over all files in the videos directory, use glob.iglob()
videos_list = glob.iglob(videos_dir + "/*.avi")

for video_path in videos_list:
    video_name = os.path.basename(video_path)

    # use subprocess to run the command
    subprocess.run(
        [
            "python3",
            "evaludate.py",
            "-d",
            "20240322-2086",
            "-arc",
            "3,3,3,3,3",
            "-c",
            "checkpoint",
            "--evaluate",
            "pretrained_h36m_detectron_coco.bin",
            "--render",
            "--viz-subject",
            video_name,
            "--viz-action",
            "custom",
            "--viz-camera",
            "0",
            "--viz-video",
            video_path,
            "--viz-export",
            video_name,
            "--viz-size",
            "6",
        ]
    )


"""
python3 evaludate.py -d 20240322-2086 -arc 3,3,3,3,3 
-c checkpoint --evaluate pretrained_h36m_detectron_coco.bin 
--render --viz-subject Banging\ Fist-30-0.avi
--viz-action custom --viz-camera 0 
--viz-video /home/ecs-user/VideoPose3D/videos/Banging\ Fist-30-0.avi 
--viz-export video_name --viz-size 6
"""
