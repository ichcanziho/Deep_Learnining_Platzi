import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def bb_intersection_over_union(ground_truth_bbox, predicted_bbox):
    xA = max(ground_truth_bbox[0], predicted_bbox[0])
    yA = max(ground_truth_bbox[1], predicted_bbox[1])
    xB = min(ground_truth_bbox[2], predicted_bbox[2])
    yB = min(ground_truth_bbox[3], predicted_bbox[3])

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    ground_truth_bbox_area = (ground_truth_bbox[2] - ground_truth_bbox[0] + 1) * (
                ground_truth_bbox[3] - ground_truth_bbox[1] + 1)
    predicted_bbox_area = (predicted_bbox[2] - predicted_bbox[0] + 1) * (predicted_bbox[3] - predicted_bbox[1] + 1)

    iou_ = intersection_area / float(ground_truth_bbox_area + predicted_bbox_area - intersection_area)

    return iou_


def show_bounding_boxes(truth, predicted):
    fig, ax = plt.subplots(figsize=(15, 15))

    ax.imshow(image)
    rect = patches.Rectangle(tuple(truth[:2]), truth[2] - truth[0], truth[3] - truth[1],
                             linewidth=3, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    rect = patches.Rectangle(tuple(predicted[:2]), predicted[2] - predicted[0], predicted[3] - predicted[1],
                             linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig("iou.png")
    plt.close()


if __name__ == '__main__':
    image = cv2.imread('mujer.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bb_truth = [680, 380, 830, 580]
    bb_predicted = [700, 400, 840, 600]

    iou = bb_intersection_over_union(bb_truth, bb_predicted)
    print(iou)

    show_bounding_boxes(bb_truth, bb_predicted)



