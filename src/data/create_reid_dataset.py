import json
import os
import shutil

import click
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#PATH_TO_DATA = "../../data/raw/NCAA_2/third_task"
#save_dir = "../../data/processed"

def copy_files(file_list, root_folder, flag):
    if len(file_list) == 0:
        return

    # Get the name of the folder
    d = os.path.split(root_folder)[1]
    print(f"Copying {len(file_list)} files from {d} to {flag} folder")
    for file in file_list:
        path = os.path.split(root_folder)[0]
        curent_folder = os.path.join(path, d)
        # Create the folder and its parent directories if they don't exist
        final_path = os.path.join(root_folder, flag)
        os.makedirs(final_path, exist_ok=True)
        shutil.move(os.path.join(curent_folder, file), final_path)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    The main function of this module is to create a dataset for the training and testing of a neural network.
    The function takes two arguments:
        1) input_filepath - The path to the folder containing all the images and annotations.
        2) output_filepath - The path where you want your new dataset to be saved.

    :param input_filepath: Specify the path to the folder containing all of your images
    :param output_filepath: Specify the path to the directory where you want to save your data
    """

    anno_pathes = []
    list_folder = []

    class_name = -13
    i = 0

    # Get all json files in subdirectories under input filepath
    for root, dirs, files in os.walk(input_filepath):
        if  len(files) != 0:
            for file in files:
                if file.endswith(".json") & (('anno' in root) or ('third_task' in root)):
                    anno_pathes.append(os.path.join(root, file))

                
    for anno_path in tqdm(anno_pathes):
        
        if 'anno' in anno_path:
            im_path = anno_path.replace("json", "jpg").replace("anno", "playerTrackingFrames")
        elif 'third_task' in anno_path:
            im_path = anno_path.replace("json", "jpg").replace("third_task", "playerTrackingFrames2")
        else:
            continue    

        im = Image.open(im_path)

        # im_w, im_h = im.size

        with open(anno_path, "r") as f:
            json_data = json.load(f)

        labels = []
        for shape in json_data['shapes']:

            label = shape['label']

            if label.isdigit():
                label = int(label)
                points = shape['points']

                folder_name = os.path.split(os.path.split(anno_path)[0])[1]

                if folder_name not in list_folder:
                    class_name += 13
                    list_folder.append(folder_name)

                save_path = os.path.join(output_filepath, f'{label + class_name}')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                x1, y1 = points[0]
                x2, y2 = points[1]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                crop_img = im.crop((x1, y1, x2, y2))
                crop_img.save(os.path.join(save_path, f'{label + class_name}' + "_c" + str(label) + "_" + str(i) + ".jpg"))
                i += 1

    root_folder = output_filepath  # Replace with the actual path to your root folder

    folders = ['train', 'test', 'valid', 'config']
    # Get a list of directories within the root folder
    directories = [name for name in os.listdir(root_folder) if (os.path.isdir(os.path.join(root_folder, name)) & (name not in folders))]

    # Count the number of directories
    num_folders = len(directories)

    print(f"Number of folders in {root_folder}: {num_folders}")    

    os.makedirs(os.path.join(root_folder, 'Market-1501'), exist_ok=True)
    root_folder = os.path.join(root_folder, 'Market-1501')

    print('Create a dataset structure!')
    for d in tqdm(directories):
        path = os.path.split(root_folder)[0]
        curent_folder = os.path.join(path, d)
        list_pathes = os.listdir(curent_folder)
        train_data, eval_data = train_test_split(list_pathes, test_size=0.1, random_state=42)
        train_data, test_data = train_test_split(train_data, test_size=0.55, random_state=42)
        
        copy_files(train_data, root_folder, 'bounding_box_train')
        copy_files(eval_data, root_folder, 'query')
        copy_files(test_data, root_folder, 'bounding_box_test')

        os.removedirs(curent_folder)
   
if  __name__ == '__main__':
    main()

