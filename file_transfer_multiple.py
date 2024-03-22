import os
from oss2_uploader import folder_downloader
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

if __name__ == "__main__":

    folder_downloader(
        "pose-daten",
        "oss-ap-southeast-1.aliyuncs.com",
        "detectron2d",
        os.path.join(
            os.path.expanduser("~"), "Documents", "VideoPose3D", "detectron2d"
        ),
    )
