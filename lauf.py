import os
import glob
from oss2_uploader import folder_downloader, folder_uploader

videos_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "videos")

# iterat over all files in the videos directory, use glob.iglob()
videos_list = glob.iglob(videos_dir + "/*.avi")

for video in videos_list:
    print(video)


"""
python3 evaludate.py -d 20240322-2086 -arc 3,3,3,3,3 
-c checkpoint --evaluate pretrained_h36m_detectron_coco.bin 
--render --viz-subject Banging\ Fist-30-0.avi
--viz-action custom --viz-camera 0 
--viz-video /home/ecs-user/VideoPose3D/videos/Banging\ Fist-30-0.avi 
--viz-export video_name --viz-size 6
"""
