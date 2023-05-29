import argparse
import glob
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

ROJECT_ROOT = '~/Projects/track_sport'

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def _iof(boxes1, boxes2):
    area1 = box_area(boxes1)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = np.clip((rb - lt), 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    return inter / area1[:, None]


def calc_max_iof_dict(labelme_dict, args):
    boxes = []
    max_iof_d = {}
    for obj in labelme_dict['shapes']:
        track_id = obj['label']
        pts = obj['points']

        if args.mode == 'ncaa':
            if not track_id.isdigit():
                continue
            track_id = int(track_id)
            if track_id not in range(1, 13):
                continue
        else:
            if 'j_' in track_id:
                continue

        xy1, xy2 = pts
        x, y = xy1
        x2, y2 = xy2
        x, y = int(x), int(y)
        x2, y2 = int(x2), int(y2)

        crop = frame[y:y2, x:x2]
        boxes.append([x, y, x2, y2])

    boxes = np.asarray(boxes)
    if len(boxes) == 0:
        return {}
    if len(boxes) == 1:
        x, y, x2, y2 = boxes[0].tolist()
        max_iof_d[(x, y, x2, y2)] = 0.
    else:
        iof_matrix = _iof(boxes, boxes)
        np.fill_diagonal(iof_matrix, 0)
        for i in range(len(boxes)):
            x, y, x2, y2 = boxes[i].tolist()
            max_iof_d[(x, y, x2, y2)] = np.max(iof_matrix[i])
    return max_iof_d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--jsons-dir',
                        type=str,
                        default='',
                        help='path to directory with jsons')
    parser.add_argument('--start-track-id',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument('--max-iof-thr',
                        type=float,
                        default=0.2,
                        help='')
    parser.add_argument('--mode',
                        type=str,
                        default='ncaa',
                        help='ncaa/baller_tv')
    parser.add_argument('--out-dir',
                        type=str,
                        default='data/NCAA/track_crops/',
                        help='path to output dir')
    args = parser.parse_args()

    json_paths = glob.glob(os.path.join(PROJECT_ROOT, args.jsons_dir) + '/**/*.json'.replace('//', '/'), recursive=True)

    track_id_counter = args.start_track_id
    track_id_mapping = {}
    glob_crop_paths = []
    crop_id = 0

    for json_p in tqdm(json_paths):
        frame_path = json_p.replace('/anno/', '/frames/').replace('.json', '.jpg')
        if not os.path.exists(frame_path):
            frame_path = frame_path.replace('.jpg', '.jpeg')

        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        img_height, img_width = frame.shape[:2]

        video_dir = os.path.dirname(json_p.replace(PROJECT_ROOT, '').replace(args.jsons_dir, '')).replace('anno/', '')
        if video_dir[0] == '/':
            video_dir = video_dir[1:]

        with open(json_p, 'r') as fp:
            labelme_dict = json.load(fp)

        max_iof_dict = calc_max_iof_dict(labelme_dict, args)

        for obj in labelme_dict['shapes']:
            track_id = obj['label']
            pts = obj['points']

            if args.mode == 'ncaa':
                if not track_id.isdigit():
                    continue
                track_id = int(track_id)
                if track_id not in range(1, 13):
                    continue
            else:
                if 'j_' in track_id:
                    continue
                track_id = int(track_id.split('_')[0])

            xy1, xy2 = pts
            x, y = xy1
            x2, y2 = xy2
            x, y = int(x), int(y)
            x2, y2 = int(x2), int(y2)

            crop = frame[y:y2, x:x2]
            if crop.shape[0] < 2 or crop.shape[1] < 2 or max_iof_dict[(x, y, x2, y2)] > args.max_iof_thr:
                continue

            if (video_dir, track_id) not in track_id_mapping:
                track_id_mapping[(video_dir, track_id)] = track_id_counter
                track_id_counter += 1

            folder_id = track_id_mapping[(video_dir, track_id)]
            crop_path = os.path.join(PROJECT_ROOT, args.out_dir, str(folder_id), f'crop_{crop_id}.jpg')
            glob_crop_paths.append(crop_path)
            os.makedirs(os.path.dirname(crop_path), exist_ok=True)
            cv2.imwrite(crop_path, crop)
            crop_id += 1

    with open(os.path.join(PROJECT_ROOT, args.out_dir, 'dataset.txt'), 'w') as f:
        for item in glob_crop_paths:
            f.write('%s\n' % item)