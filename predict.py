import cv2
from foot_utils import util
from segment_anything import SamPredictor
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


import numpy as np
import matplotlib.pyplot as plt

global foot_field
foot_field = []


def get_coeff(x1, y1, x2, y2):
    """
    Get the coefficient from a line
    :param x1: float: x value of the first line's point
    :param y1: float: y value of the first line's point
    :param x2: float: x value of the second line's point
    :param y2: float: y value of the second line's point
    :return:
    """
    coeff = np.polyfit([x1, x2], [y1, y2], 1)
    a = coeff[0]
    b = coeff[1]
    return a, b


def get_boxe_in_field(list_boxes, field):
    lf = field[0]
    rf = field[1]
    tf = field[2]
    bf = field[3]
    boxes_in_field = []
    for boxe in list_boxes:
        x1, y1, x2, y2 = boxe
        central = [x1 + ((x2 - x1) / 2), y2]
        al, bl = get_coeff(lf[0][0], lf[0][1], lf[1][0], lf[1][1])
        ar, br = get_coeff(rf[0][0], rf[0][1], rf[1][0], rf[1][1])
        if central[0] >= (central[1] - bl) / al and central[0] <= (central[1] - br) / ar:
            if central[1] > min(tf[0][1], tf[1][1]) and central[1] < max(bf[0][1], bf[1][1]):
                boxes_in_field.append(boxe)

    return boxes_in_field


def click_event(event, x, y, flags, param):
    # checking for left mouse clicks

    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        foot_field.append([x, y])


def get_field_by_asking(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 1920, 1080)
    cv2.imshow('image', img)
    print("First clicko the point for the left line ")
    print("Second click on the points for the reight line")
    print("Third click on th epoints for the top line")
    print("Fourth click on the points for bottom line")

    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    left_line = foot_field[0:2]
    right_line = foot_field[2:4]
    top_line = foot_field[4:6]
    bottom_line = foot_field[6:8]
    return [left_line, right_line, top_line, bottom_line]


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def draw_masks_fromDict(image, masks_generated) :
  masked_image = image.copy()
  for i in range(len(masks_generated)) :
    masked_image = np.where(np.repeat(masks_generated[i]['segmentation'].astype(int)[:, :, np.newaxis], 3, axis=2),
                            np.random.choice(range(256), size=3),
                            masked_image)

    masked_image = masked_image.astype(np.uint8)

  return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)


sam = sam_model_registry["vit_l"](checkpoint="segment-anything/checkpoints/sam_vit_l_0b3195.pth")
#image = cv2.imread("/home/reda/Documents/projets/results_detectron/rosen/DJI_0019/images/frame_246.jpg")
video_path = '/home/reda/Documents/projets/drone/rosen_esquelbeck_10_09/DJI_0019.MP4'

# Open the video file
cap = cv2.VideoCapture(video_path)
ret, image = cap.read()

print(image)

# [left_line, right_line, top_line, bottom_line] = get_field_by_asking(image)

sam.to(device='cuda')
mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=80, pred_iou_thresh=0.75)
predictor = SamPredictor(sam)
masks = mask_generator.generate(image)
sorted_masks = sorted(masks, key=lambda x: x['area'])
segmented_image = draw_masks_fromDict(image, sorted_masks[:45])
cv2.imshow('seg',segmented_image)