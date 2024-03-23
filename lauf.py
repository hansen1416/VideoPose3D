import os
from oss2_uploader import folder_downloader, folder_uploader


"""
python3 evaludate.py -d 20240322-2086 -arc 3,3,3,3,3 
-c checkpoint --evaluate pretrained_h36m_detectron_coco.bin 
--render --viz-subject Banging\ Fist-30-0.avi
--viz-action custom --viz-camera 0 
--viz-video /home/ecs-user/VideoPose3D/videos/Banging\ Fist-30-0.avi 
--viz-export video_name --viz-size 6
"""
