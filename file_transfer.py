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


if __name__ == "__main__":

    folder_downloader(
        "pose-daten",
        "oss-ap-southeast-1.aliyuncs.com",
        "videos",
        os.path.join(os.path.expanduser("~"), "VideoPose3D", "videos"),
    )
