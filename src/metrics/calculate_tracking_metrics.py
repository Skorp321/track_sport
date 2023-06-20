# import required packages
import motmetrics as mm
import numpy as np
import argparse


def motMetricsEnhancedCalculator(gtSource, tSource):
    # load ground truth
    gt = np.loadtxt(gtSource, delimiter=" ")

    # load tracking output
    t = np.loadtxt(tSource, delimiter=" ")

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1

        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        gt_dets = gt[gt[:, 0] == frame, 1:6]  # select all detections in gt
        t_dets = t[t[:, 0] == frame, 1:6]  # select all detections in t

        C = mm.distances.iou_matrix(
            gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5
        )  # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(
            gt_dets[:, 0].astype("int").tolist(), t_dets[:, 0].astype("int").tolist(), C
        )

    mh = mm.metrics.create()

    summary = mh.compute(
        acc,
        metrics=["num_frames", "idf1", "recall", "precision", "mota", "motp"],
        name="acc",
    )

    strsummary = mm.io.render_summary(
        summary,
        # formatters={'mota' : '{:.2%}'.format},
        namemap={
            "idf1": "IDF1",
            "recall": "Rcll",
            "precision": "Prcn",
            "mota": "MOTA",
            "motp": "MOTP",
        },
    )
    print(strsummary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gtSource", type=str, help="Enter your name")
    parser.add_argument("--tSource", type=str, help="Enter your age", default=0)

    args = parser.parse_args()

    motMetricsEnhancedCalculator(args.gtSource, args.tSource)
