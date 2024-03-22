import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


def folder_downloader(bucket_name, oss_endpoint, oss_prefix, target_dir):

    # 使用环境变量中获取的RAM用户的访问密钥配置访问凭证。
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

    # yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
    # 填写Bucket名称，并设置连接超时时间为30秒。
    bucket = oss2.Bucket(auth, oss_endpoint, bucket_name, connect_timeout=30)

    if oss_prefix[-1] != "/":
        oss_prefix += "/"

    # list all files in the oss_prefix
    osskey_list = [obj.key for obj in oss2.ObjectIterator(bucket, prefix=oss_prefix)]

    # get cpu count

    print(f"Downloading {len(osskey_list)} files from {bucket_name}")

    for i, osskey in enumerate(osskey_list):
        # split `osskey` into `path` and `filename`
        filename = osskey.split("/")[-1]

        target_path = os.path.join(target_dir, filename)

        if os.path.isfile(target_path):
            print(f"File {filename} already exists, skipping {i}/{len(osskey_list)}")
            continue

        if os.path.isfile(target_path):
            continue

        bucket.get_object_to_file(osskey, target_path)

        print(f"Downloaded {osskey} to {target_path} {i}/{len(osskey_list)}")


def folder_uploader(folder_path, bucket_name, oss_endpoint, oss_prefix):

    # 使用环境变量中获取的RAM用户的访问密钥配置访问凭证。
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

    # yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
    # 填写Bucket名称，并设置连接超时时间为30秒。
    bucket = oss2.Bucket(auth, oss_endpoint, bucket_name, connect_timeout=30)

    folder_path = os.path.abspath(folder_path)

    if not os.path.exists(folder_path):
        print(f"{folder_path} does not exist")
        return

    # get all files in the folder
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

    osskeys = set([obj.key for obj in oss2.ObjectIterator(bucket, prefix=oss_prefix)])

    for i, filepath in enumerate(all_files):
        if not os.path.isfile(filepath):
            continue

        filename = os.path.basename(filepath)

        if oss_prefix[-1] != "/":
            oss_prefix += "/"

        target_path = f"{oss_prefix}{filename}"

        # check if the file already exists in oss
        if target_path in osskeys:
            print(
                f"{target_path} already exists in oss, skipping. {i}/{len(all_files)}"
            )
            continue

        bucket.put_object_from_file(
            f"{target_path}",
            filepath,
        )

        print(f"{target_path} uploaded {i}/{len(all_files)}")


if __name__ == "__main__":

    # folder_downloader(
    #     "pose-daten",
    #     "oss-ap-southeast-1.aliyuncs.com",
    #     "videos",
    #     os.path.join(os.path.expanduser("~"), "VideoPose3D", "videos"),
    # )

    # folder_uploader(
    #     os.path.join(os.path.expanduser("~"), "VideoPose3D", "detectron2d"),
    #     "pose-daten",
    #     "oss-ap-southeast-1.aliyuncs.com",
    #     "detectron2d",
    # )

    folder_uploader(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_dataset"),
        "pose-daten",
        "oss-ap-southeast-1.aliyuncs.com",
        "custom_dataset",
    )

    folder_downloader(
        "pose-daten",
        "oss-ap-southeast-1.aliyuncs.com",
        "custom_dataset",
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_dataset"),
    )
