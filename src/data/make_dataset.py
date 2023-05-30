# -*- coding: utf-8 -*-
import json
import os
import shutil
from tqdm import tqdm

# import click
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
#from dotenv import find_dotenv, load_dotenv


def make_pathes_list_jpg(input_dirs: list):
    """
    Функция создает список путей к файлам формата jpg в заданных директориях input_dirs
    и записывает их в csv-файлы в директории 'sport/data/interim'.

    Аргументы:
    :param input_dirs (list): список путей к директориям с изображениями в формате jpg.

    Возвращает:
    Ничего не возвращает.
    """
    anno_pathes_jpg = []
    for input_dir in input_dirs:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                # проверяем, есть ли заданный файл в списке файлов текущей папки
                if "jpg" in file:
                    # если есть, то добавляем путь в список
                    anno_pathes_jpg.append(os.path.join(root, file))

    df = pd.DataFrame(anno_pathes_jpg)
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df, eval_df = train_test_split(train_df, test_size=0.2)
    train_df.to_csv(
        os.path.join(
            "./data", "ncaa_train.csv"
        ),
        header=False,
        index=False,
    )
    eval_df.to_csv(
        os.path.join(
            "./data", "ncaa_valid.csv"
        ),
        header=False,
        index=False,
    )
    test_df.to_csv(
        os.path.join(
            "./data", "ncaa_test.csv"
        ),
        header=False,
        index=False,
    )


def copy_jpg_fiels():
    input_dir = "./data"
    output_dir = "./data"

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if "train" in file:
                print("Create train folder with images!")
                train_name = "train//images"

                if not os.path.exists(os.path.join(output_dir, train_name)):
                    if not os.path.exists(os.path.join(output_dir, "train")):
                        os.makedirs(os.path.join(output_dir, "train"))
                    os.makedirs(os.path.join(output_dir, train_name))

                csv_path = os.path.join(input_dir, file)
                dataframe = pd.read_csv(csv_path, header=None)

                for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
                    path, file_name = os.path.split(row[0])
                    _, folder = os.path.split(path)
                    file_name = folder + "-" + file_name

                    try:
                        shutil.copyfile(
                            row[0], os.path.join(output_dir, train_name, file_name)
                        )
                    except Exception as e:
                        print(f"Ошибка при копировании файла {row}: {e}")
                print()

            elif "valid" in file:
                print("Create valid folder with images!")
                eval_name = "valid//images"

                if not os.path.exists(os.path.join(output_dir, eval_name)):
                    os.makedirs(os.path.join(output_dir, "valid"))
                    os.makedirs(os.path.join(output_dir, eval_name))

                csv_path = os.path.join(input_dir, file)
                dataframe = pd.read_csv(csv_path, header=None)

                for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
                    path, file_name = os.path.split(row[0])
                    _, folder = os.path.split(path)
                    file_name = folder + "-" + file_name

                    try:
                        shutil.copyfile(
                            row[0], os.path.join(output_dir, eval_name, file_name)
                        )
                    except Exception as e:
                        print(f"Ошибка при копировании файла {row}: {e}")
                print()

            elif "test" in file:
                print("Create test folder with images!")
                test_name = "test//images"

                if not os.path.exists(os.path.join(output_dir, test_name)):
                    os.makedirs(os.path.join(output_dir, "test"))
                    os.makedirs(os.path.join(output_dir, test_name))

                csv_path = os.path.join(input_dir, file)
                dataframe = pd.read_csv(csv_path, header=None)

                for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
                    path, file_name = os.path.split(row[0])
                    _, folder = os.path.split(path)
                    file_name = folder + "-" + file_name

                    try:
                        shutil.copyfile(
                            row[0], os.path.join(output_dir, test_name, file_name)
                        )
                    except Exception as e:
                        print(f"Ошибка при копировании файла {row}: {e}")
                print()


def create_annotations():
    input_dir = "./data"
    output_dir = "./data"

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if "train" in file:
                print("Create train folder with annotations!")
                train_name = "train//labels"

                if not os.path.exists(os.path.join(output_dir, train_name)):
                    if not os.path.exists(os.path.join(output_dir, "train")):
                        os.makedirs(os.path.join(output_dir, "train"))
                    os.makedirs(os.path.join(output_dir, train_name))

                csv_path = os.path.join(input_dir, file)
                dataframe = pd.read_csv(csv_path, header=None)

                for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
                    path, file_name = os.path.split(row[0])

                    if "playerTrackingFrames2" in path:
                        path = path.replace("playerTrackingFrames2", "third_task")
                    elif "playerTrackingFrames" in path:
                        path = path.replace("playerTrackingFrames", "anno")

                    _, folder = os.path.split(path)
                    file_name_long = folder + "-" + file_name
                    file_name = file_name.replace("jpg", "json")

                    anno_path = os.path.join(path, file_name)

                    if not os.path.exists(anno_path):
                        continue

                    with open(anno_path, "r") as f:
                        data = json.load(f)

                    # Создание списка строк с аннотациями для каждого изображения
                    annotations = []
                    for shape in data["shapes"]:
                        if shape["label"].isdigit():
                            # Извлечение координат прямоугольника
                            x1, y1 = shape["points"][0]
                            x2, y2 = shape["points"][1]
                            # Вычисление центра объекта и его размеров
                            x_center = (x1 + x2) / 2 / data["imageWidth"]
                            y_center = (y1 + y2) / 2 / data["imageHeight"]
                            width = (x2 - x1) / data["imageWidth"]
                            height = (y2 - y1) / data["imageHeight"]
                            # Добавление строки аннотации в список
                            annotations.append(
                                f"0 {x_center} {y_center} {width} {height}"
                            )

                    file_name_long = file_name_long.replace("jpg", "txt")
                    # Запись аннотаций в файл
                    with open(
                        os.path.join(output_dir, train_name, file_name_long), "w"
                    ) as f:
                        f.write("\n".join(annotations))

                print()

            elif "valid" in file:
                print("Create eval folder with annotations!")
                eval_name = "valid//labels"

                if not os.path.exists(os.path.join(output_dir, eval_name)):
                    if not os.path.exists(os.path.join(output_dir, "valid")):
                        os.makedirs(os.path.join(output_dir, "valid"))
                    os.makedirs(os.path.join(output_dir, eval_name))

                csv_path = os.path.join(input_dir, file)
                dataframe = pd.read_csv(csv_path, header=None)

                for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
                    path, file_name = os.path.split(row[0])

                    if "playerTrackingFrames2" in path:
                        path = path.replace("playerTrackingFrames2", "third_task")
                    elif "playerTrackingFrames" in path:
                        path = path.replace("playerTrackingFrames", "anno")

                    _, folder = os.path.split(path)
                    file_name_long = folder + "-" + file_name
                    file_name = file_name.replace("jpg", "json")

                    anno_path = os.path.join(path, file_name)

                    if not os.path.exists(anno_path):
                        continue

                    with open(anno_path, "r") as f:
                        data = json.load(f)

                    # Создание списка строк с аннотациями для каждого изображения
                    annotations = []
                    for shape in data["shapes"]:
                        if shape["label"].isdigit():
                            # Извлечение координат прямоугольника
                            x1, y1 = shape["points"][0]
                            x2, y2 = shape["points"][1]
                            # Вычисление центра объекта и его размеров
                            x_center = (x1 + x2) / 2 / data["imageWidth"]
                            y_center = (y1 + y2) / 2 / data["imageHeight"]
                            width = (x2 - x1) / data["imageWidth"]
                            height = (y2 - y1) / data["imageHeight"]
                            # Добавление строки аннотации в список
                            annotations.append(
                                f"0 {x_center} {y_center} {width} {height}"
                            )

                    file_name_long = file_name_long.replace("jpg", "txt")
                    # Запись аннотаций в файл
                    with open(
                        os.path.join(output_dir, eval_name, file_name_long), "w"
                    ) as f:
                        f.write("\n".join(annotations))
                print()

            elif "test" in file:
                print("Create test folder with annotations!")
                test_name = "test//labels"

                if not os.path.exists(os.path.join(output_dir, test_name)):
                    if not os.path.exists(os.path.join(output_dir, "test")):
                        os.makedirs(os.path.join(output_dir, "test"))
                    os.makedirs(os.path.join(output_dir, test_name))

                csv_path = os.path.join(input_dir, file)
                dataframe = pd.read_csv(csv_path, header=None)

                for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
                    path, file_name = os.path.split(row[0])

                    if "playerTrackingFrames2" in path:
                        path = path.replace("playerTrackingFrames2", "third_task")
                    elif "playerTrackingFrames" in path:
                        path = path.replace("playerTrackingFrames", "anno")

                    _, folder = os.path.split(path)
                    file_name_long = folder + "-" + file_name
                    file_name = file_name.replace("jpg", "json")

                    anno_path = os.path.join(path, file_name)

                    if not os.path.exists(anno_path):
                        continue

                    with open(anno_path, "r") as f:
                        data = json.load(f)

                    # Создание списка строк с аннотациями для каждого изображения
                    annotations = []
                    for shape in data["shapes"]:
                        if shape["label"].isdigit():
                            # Извлечение координат прямоугольника
                            x1, y1 = shape["points"][0]
                            x2, y2 = shape["points"][1]
                            # Вычисление центра объекта и его размеров
                            x_center = (x1 + x2) / 2 / data["imageWidth"]
                            y_center = (y1 + y2) / 2 / data["imageHeight"]
                            width = (x2 - x1) / data["imageWidth"]
                            height = (y2 - y1) / data["imageHeight"]
                            # Добавление строки аннотации в список
                            annotations.append(
                                f"0 {x_center} {y_center} {width} {height}"
                            )

                    file_name_long = file_name_long.replace("jpg", "txt")
                    # Запись аннотаций в файл
                    with open(
                        os.path.join(output_dir, test_name, file_name_long), "w"
                    ) as f:
                        f.write("\n".join(annotations))
                print()


if __name__ == "__main__":
    paths = [
        "/media/skorp/datasets/PycharmProjects/MLOps_sport/sport/data/raw/NCAA/NCAA_1_tracking",
        "/media/skorp/datasets/PycharmProjects/MLOps_sport/sport/data/raw/NCAA/NCAA_2",
    ]

    make_pathes_list_jpg(paths)
    copy_jpg_fiels()
    create_annotations()
