import json
import os
import shutil
import click
import numpy as np
from tqdm import tqdm

@click.command()
@click.argument('folder_path', type=click.Path(exists=True))
def create_test_dataset(folder_path):

    num = 0

    folders_list = os.listdir(folder_path)
    folders_list.sort()

    for folder in folders_list:
        num += 1

        print(f'Копирование папки {folder}')
        store_path_img = f'data\\ncaa_dataset\\images'
        os.makedirs(store_path_img, exist_ok=True)

        store_path_annos = f'data\\ncaa_dataset\\annos'
        os.makedirs(store_path_annos, exist_ok=True)

        full_folder_path = os.path.join(folder_path, folder)
        img_list = os.listdir(full_folder_path)

        full_anno_folder_path = full_folder_path.replace('images', 'anno')
        
        for img in tqdm(img_list):

            img_path = os.path.join(full_folder_path, img)
            shutil.copy(img_path, os.path.join(store_path_img, img))

            anno = img.replace('jpg', 'json')
            anno_path = os.path.join(full_anno_folder_path, anno)
            
            if os.path.exists(anno_path):
                shutil.copy(anno_path, os.path.join(store_path_annos, anno))

    create_gt_file()            

def create_gt_file():
    imgs_paths = 'data\\ncaa_dataset\\images'

    list_images = os.listdir(imgs_paths)
    list_images.sort()
    print()
    print('Создаем gt_file')

    with open(f'data\\ncaa_dataset\\gt_xywh.txt', 'w') as output_file:
        for i, img in enumerate(tqdm(list_images), start=1):
            anno = img.replace('jpg', 'json')
            anno_path = os.path.join('data/ncaa_dataset/annos', anno)

            if not os.path.exists(anno_path):
                continue

            with open(anno_path, "r") as f:
                data = json.load(f)

            for shape in data['shapes']:
                label = shape['label']
                if label.isdigit():
                    # Извлечение координат прямоугольника
                    x1, y1 = shape["points"][0]
                    x2, y2 = shape["points"][1]

                    x1 = int(x1)
                    y1 = int(y1)
                    width = int(np.abs((x2 - x1)))
                    height = int(np.abs((y2 - y1)))

                    line = f"{i} {label} {x1:.0f} {y1:.0f} {width:.0f} {height:.0f} -1 -1 -1 0\n"

                    output_file.write(line)


if __name__ == '__main__':
    create_test_dataset()
