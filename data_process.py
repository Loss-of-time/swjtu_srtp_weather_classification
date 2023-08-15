from utils import delete_folder_contents

import os
import random
import shutil
from pathlib import Path

import requests
from PIL import Image
from loguru import logger


def backup_folder(source_folder: Path, backup_folder: Path) -> None:
    try:
        # 使用 shutil.copytree() 复制文件夹及其内容
        shutil.copytree(source_folder, backup_folder)
        logger.info("文件夹备份成功！")
    except Exception as e:
        logger.info("文件夹备份失败:", e)


class ImageOperation():
    @staticmethod
    def min_dimension(image_path: Path | str, min_dimension: int):
        image = Image.open(image_path)
        # 获取原始图片的宽度和高度
        width, height = image.size
        # 计算新的宽度和高度，保持长宽比
        if width > height:
            new_height = min_dimension
            new_width = int(width * (min_dimension / height))
        else:
            new_width = min_dimension
            new_height = int(height * (min_dimension / width))
            # 缩放图片
        resized_image = image.resize((new_width, new_height))
        return resized_image

    @staticmethod
    def center_crop(image_path: Path | str, target_size: int):
        image = Image.open(image_path)
        # 获取原始图片的宽度和高度
        width, height = image.size
        # 计算裁剪框的位置
        left = (width - target_size) / 2
        top = (height - target_size) / 2
        right = (width + target_size) / 2
        bottom = (height + target_size) / 2
        # 进行中心裁剪
        cropped_image = image.crop(
            (int(left), int(top), int(right), int(bottom)))
        return cropped_image


class PathOperation():
    @staticmethod
    def apply_label(path: Path, label):
        with open(label, 'r') as f:
            labels = f.readlines()

        for i in labels:
            if i[-1] == '\n':
                i = i[:-1]
            file, t = tuple(i.split('->'))
            file_name, suffix = tuple(file.split('.'))
            os.rename(path/file, path/'.'.join([file_name, t, suffix]))

    @staticmethod
    def delete_d(path: Path):
        for i in os.listdir(path):
            if i.split('.')[-2] == 'd':
                os.remove(path / i)
                logger.info(f'{path / i} deleted.')

    @staticmethod
    def reshape(path: Path, w, h):
        for i in os.listdir(path):
            image = Image.open(path / i)
            image = image.resize((w, h))
            image.save(path / i)
            logger.info('{} reshaped'.format(path / i))

    @staticmethod
    def center_crop_images(input_path: Path, target: int):
        for input in os.listdir(input_path):
            image_path = input_path / input
            cropped_image = ImageOperation.center_crop(image_path, target)
            cropped_image.save(image_path)
            logger.info("{} center croped.".format(image_path))

    @staticmethod
    def add_type(path: Path, type: str):
        for i in os.listdir(path):
            sp = i.split('.')
            if type == '':
                sp[-2] = sp[-1]
                sp = sp[:-1]
            else:
                sp.append(sp[-1])
                sp[-2] = type
            raw = path / i
            tar = path / '.'.join(sp)
            os.rename(raw, tar)
            logger.info("{} -> {}".format(raw, tar))

    @staticmethod
    def split_data(raw: Path, path_a: Path, path_b: Path, a_percent: float) -> None:
        delete_folder_contents(path_a)
        delete_folder_contents(path_b)
        files = os.listdir(raw)
        files = random.sample(files, len(files))
        split_point = int(len(files) * a_percent)
        for i in range(0, split_point):
            shutil.copyfile(raw / files[i], path_a / files[i])
            logger.info('{} -> {}'.format(raw / files[i], path_a / files[i]))
        for i in range(split_point, len(files)):
            shutil.copyfile(raw / files[i], path_b / files[i])
            logger.info('{} -> {}'.format(raw / files[i], path_b / files[i]))
