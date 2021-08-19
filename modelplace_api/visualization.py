import os
from collections import defaultdict
from typing import Generator, List

import numpy as np
from loguru import logger

from .colors import RGBA_COLORS
from .objects import (
    AgeGenderLabel,
    BBox,
    CountableVideoFrame,
    EmotionLabel,
    Label,
    Landmarks,
    Mask,
    Point,
    Pose,
    TextPolygon,
    VideoFrame,
)
from .utils import decode_coco_rle

try:
    import cv2
    import imageio
    import skvideo
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    logger.warning(
        "Some dependencies is invalid. "
        "Please install this package with extra requiements. "
        "For unix: pip install modelplace-api[vis] "
        "For windows: pip install modelplace-api[vis-windows]",
    )


FFMPEG_OUTPUT_DICT = {
    "-vcodec": "libx264",
    "-vf": "format=yuv420p",
    "-movflags": "+faststart",
}
text_style_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "text_styles")

MONTSERATT_BOLD_TTF_PATH = os.path.join(text_style_dir, "Montserrat-Bold.ttf")
MONTSERATT_REGULAR_TTF_PATH = os.path.join(text_style_dir, "Montserrat-Regular.ttf")
MONTSERATT_MEDIUM_TTF_PATH = os.path.join(text_style_dir, "Montserrat-Medium.ttf")

BACKGROUND_COLOR = (79, 79, 79, 1)
CORALL_COLOR = (236, 113, 108, 1)
DELIMITER_COLOR = (255, 255, 255, 1)
WHITE_TEXT_COLOR = (255, 255, 255, 1)
BLACK_TEXT_COLOR = (0, 0, 0, 1)
LAVANDER_COLOR = (125, 211, 210, 1)
DARK_PINK_COLOR = (109, 84, 199, 1)
NORM_HEIGHT = 591
CLASS_BOX_WIDTH = 80
CLASS_BOX_HEIGHT = 40
INFO_BOX_WIDTH = 60
INFO_BOX_HEIGHT = 30
PROB_BOX_HEIGHT = PROB_BOX_WIDTH = 70
TEXT_OFFSET_X = 16
TEXT_OFFSET_Y = 8
BOX_CORNER_OFFSET = 24
CLASS_BOX_CORNER_OFFSET = 13
CLASS_BOX_TEXT_OFFSET_Y = 12
CLASS_BOX_TEXT_OFFSET_X = 18
COMMENT_TEXT_SIZE = 16
PROB_DIGIT_TEXT_SIZE = 22
PROB_DIGIT_OFFSET_Y = 14
PROB_COMMENT_OFFSET_Y = 8
PROB_COMMENT_OFFSET_X = 10
INFO_TEXT_SIZE = 12
CLASS_TEXT_SIZE = 14
LINE_THINKNESS = 3
POSE_EDGE_THINKNESS = 6
POSE_POINT_SIZE = 10
MESH_POINT_SIZE = 1.5


def get_text_width(image, text, text_size, ttf_path):
    return ImageDraw.Draw(Image.fromarray(image)).textsize(
        text, font=ImageFont.truetype(ttf_path, text_size),
    )[0]


def add_text(
    image: np.ndarray,
    text: str,
    coords: list,
    text_size: int,
    text_color: tuple,
    ttf_path: str,
    box_w: int,
) -> np.ndarray:
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    montserrat = ImageFont.truetype(ttf_path, text_size)
    text_w = get_text_width(image, text, text_size, ttf_path)
    coords[0] = coords[0] + int((box_w - text_w) / 2)
    draw.text(coords, text, font=montserrat, fill=text_color)
    return np.array(pil_img)


def add_probability_box(
    image: np.ndarray,
    probability: str,
    text: str,
    coords: List,
    delimiter: bool = False,
    box_color: tuple = BACKGROUND_COLOR,
) -> np.ndarray:
    img_h, img_w, _ = image.shape
    scale = min([img_w, img_h]) / NORM_HEIGHT
    prob_size = int(scale * PROB_DIGIT_TEXT_SIZE)
    text_size = int(scale * COMMENT_TEXT_SIZE)
    box_x1, box_y1, box_x2, box_y2 = coords
    box_w = box_x2 - box_x1
    box_h = box_y2 - box_y1
    text_offset_y1 = int(scale * PROB_DIGIT_OFFSET_Y)
    text_offset_y2 = int(box_h - scale * (COMMENT_TEXT_SIZE + 10))

    image = cv2.rectangle(
        image, (box_x1, box_y1), (box_x2, box_y2), box_color, thickness=-1,
    )
    image = add_text(
        image,
        probability,
        [box_x1, box_y1 + text_offset_y1],
        prob_size,
        WHITE_TEXT_COLOR,
        MONTSERATT_BOLD_TTF_PATH,
        box_w,
    )
    image = add_text(
        image,
        text,
        [box_x1, box_y1 + text_offset_y2],
        text_size,
        WHITE_TEXT_COLOR,
        MONTSERATT_REGULAR_TTF_PATH,
        box_w,
    )
    if delimiter:
        line_offset_x = int(scale * TEXT_OFFSET_X)
        image = cv2.line(
            image,
            (box_x1 + line_offset_x, box_y2),
            (box_x2 - line_offset_x, box_y2),
            DELIMITER_COLOR,
            1,
            cv2.LINE_AA,
        )
    return image


def add_class_box(
    image: np.ndarray,
    background_color: tuple,
    text: str,
    box_w: int,
    box_number: int = 0,
) -> np.ndarray:
    img_h, img_w, _ = image.shape
    scale = min([img_w, img_h]) / NORM_HEIGHT
    text_size = int(scale * CLASS_TEXT_SIZE)
    box_offset_x = box_offset_y = int(scale * CLASS_BOX_CORNER_OFFSET)
    box_h = int(scale * CLASS_BOX_HEIGHT)
    text_offset_y = int(scale * CLASS_BOX_TEXT_OFFSET_Y)

    box_x2 = img_w - box_offset_x
    box_x1 = box_x2 - box_w
    box_y2 = box_offset_y + (1 + box_number) * box_h
    box_y1 = box_y2 - box_h
    image = cv2.rectangle(
        image, (box_x1, box_y1), (box_x2, box_y2 - 2), background_color, thickness=-1,
    )
    image = add_text(
        image,
        text,
        [box_x1, box_y1 + text_offset_y],
        text_size,
        WHITE_TEXT_COLOR,
        MONTSERATT_MEDIUM_TTF_PATH,
        box_w,
    )
    return image


def add_bbox(image: np.ndarray, coords: List, box_color: tuple) -> np.ndarray:
    (x1, y1, x2, y2) = coords
    img_h, img_w, _ = image.shape
    scale = min([img_w, img_h]) / NORM_HEIGHT
    thickness = int(scale * LINE_THINKNESS)
    image = cv2.rectangle(
        image, (x1, y1), (x2, y2), box_color, thickness=thickness, lineType=cv2.LINE_AA,
    )
    return image


def add_info(
    image: np.ndarray, coords: List, box_color: tuple, text: str, text_color: tuple,
) -> np.ndarray:
    img_h, img_w, _ = image.shape
    scale = min([img_w, img_h]) / NORM_HEIGHT
    coords = coords[0] - int(2 * scale), coords[1] + int(2 * scale)
    text_size = int(scale * INFO_TEXT_SIZE)
    box_w = int(scale * INFO_BOX_WIDTH)
    box_h = int(scale * INFO_BOX_HEIGHT)
    text_offset_y = int(scale * TEXT_OFFSET_Y)
    text_w = get_text_width(image, text, text_size, MONTSERATT_BOLD_TTF_PATH)
    box_w = max(box_w, text_w + text_offset_y * 2)
    image = cv2.rectangle(
        image,
        (coords[0], coords[1] - box_h),
        (coords[0] + box_w, coords[1]),
        box_color,
        thickness=-1,
    )
    image = add_text(
        image,
        text,
        [coords[0], coords[1] - box_h + text_offset_y],
        text_size,
        text_color,
        MONTSERATT_BOLD_TTF_PATH,
        box_w,
    )
    return image


def add_pose(image: np.ndarray, detection: Pose, edge_color: tuple) -> np.ndarray:
    img_h, img_w, _ = image.shape
    scale = min([img_w, img_h]) / NORM_HEIGHT
    edge_thinkness = int(scale * POSE_EDGE_THINKNESS)
    for link in detection.links:
        if (
            link.joint_b.x == link.joint_b.y == 0
            or link.joint_a.x == link.joint_a.y == 0
        ):
            continue

        cv2.line(
            image,
            (int(link.joint_a.x), int(link.joint_a.y)),
            (int(link.joint_b.x), int(link.joint_b.y)),
            edge_color,
            edge_thinkness,
        )

    joints = [
        joint for link in detection.links for joint in [link.joint_a, link.joint_b]
    ]
    unique_joints = [
        a
        for i, a in enumerate(joints)
        if not any(a.class_name == b.class_name for b in joints[:i])
    ]
    for joint in unique_joints:
        if joint.x == joint.y == 0:
            continue
        image = add_point(image, joint, LAVANDER_COLOR, POSE_POINT_SIZE)
    return image


def add_point(
    image: np.ndarray, point: Point, color: tuple, point_size: int = 4,
) -> np.ndarray:
    img_h, img_w, _ = image.shape
    scale = min([img_w, img_h]) / NORM_HEIGHT
    cv2.circle(
        image, (int(point.x), int(point.y)), int(scale * point_size), color, -1,
    )
    return image


def add_poly(
    image: np.ndarray, poly: TextPolygon, color: tuple, thinkness: int,
) -> np.ndarray:
    img_h, img_w, _ = image.shape
    scale = min([img_w, img_h]) / NORM_HEIGHT
    thinkness = int(scale * thinkness)
    points = np.array([[point.x, point.y] for point in poly.points], dtype=np.int32)
    image = cv2.polylines(image, [points], True, color, thinkness)
    return image


def add_mask(image: np.ndarray, idx: np.ndarray, color: tuple) -> np.ndarray:
    mask = np.zeros_like(image).astype(np.uint8)
    mask[idx] = color[: mask.shape[2]]
    alpha = 0.8
    beta = 0.4
    image = alpha * image + beta * mask
    image = image * (2 - np.max(image) / 255)
    return image.astype(np.uint8)


def add_legend(
    image: np.ndarray, classes: List, picked_color: tuple, picked_class_number: int,
) -> np.ndarray:

    img_h, img_w, _ = image.shape
    scale = min([img_w, img_h]) / NORM_HEIGHT
    text_size = int(scale * CLASS_TEXT_SIZE)
    text_offset_x = int(scale * CLASS_BOX_TEXT_OFFSET_X)
    box_w = max(
        [int(scale * CLASS_BOX_WIDTH)]
        + [
            get_text_width(
                image, class_name.capitalize(), text_size, MONTSERATT_MEDIUM_TTF_PATH,
            )
            + text_offset_x * 2
            for class_name in classes
        ],
    )
    for class_number, class_name in enumerate(classes):
        color = (
            picked_color if class_number == picked_class_number else BACKGROUND_COLOR
        )
        image = add_class_box(
            image, color, class_name.capitalize(), box_w, class_number,
        )
    return image


def add_legend_all_classes(image: np.ndarray, classes: List) -> np.ndarray:
    img_h, img_w, _ = image.shape
    scale = min([img_w, img_h]) / NORM_HEIGHT
    text_size = int(scale * CLASS_TEXT_SIZE)
    text_offset_x = int(scale * CLASS_BOX_TEXT_OFFSET_X)
    box_w = max(
        [int(scale * CLASS_BOX_WIDTH)]
        + [
            get_text_width(
                image, class_name.capitalize(), text_size, MONTSERATT_MEDIUM_TTF_PATH,
            )
            + text_offset_x * 2
            for class_name in classes
        ],
    )
    for class_number, class_name in enumerate(classes):
        image = add_class_box(
            image,
            RGBA_COLORS[class_number],
            class_name.capitalize(),
            box_w,
            class_number,
        )
    return image


def draw_detections_one_frame(image: np.ndarray, detections: List[BBox]) -> np.ndarray:
    classes = defaultdict(list)
    for det in detections:
        classes[det.class_name].append(det)
    for class_number, class_detections in enumerate(classes.values()):
        color = RGBA_COLORS[class_number]
        for detection in class_detections:
            image = add_bbox(
                image, [detection.x1, detection.y1, detection.x2, detection.y2], color,
            )
    image = add_legend_all_classes(image, classes)
    return image


def draw_detections(image: np.ndarray, detections: List[BBox]) -> List[np.ndarray]:
    source_image = image.copy()
    images = []
    classes = defaultdict(list)
    for det in detections:
        classes[det.class_name].append(det)
    for class_number, class_detections in enumerate(classes.values()):
        one_class_image = source_image.copy()
        color = RGBA_COLORS[class_number]
        for detection in class_detections:
            one_class_image = add_bbox(
                one_class_image,
                [detection.x1, detection.y1, detection.x2, detection.y2],
                color,
            )
        one_class_image = add_legend(one_class_image, classes, color, class_number)
        images.append(one_class_image)
    images.append(add_legend(source_image, classes, BACKGROUND_COLOR, -1))
    return images


def draw_segmentation_one_frame(image: np.ndarray, detection: Mask) -> np.ndarray:
    classes = [
        class_name
        for class_number, class_name in enumerate(detection.classes)
        if class_number in detection.mask["classes"]
    ]
    for class_number, rle_mask in enumerate(detection.mask["binary"]):
        color = RGBA_COLORS[class_number]
        decoded_mask = decode_coco_rle(rle_mask)
        idx = decoded_mask == 1
        image = add_mask(image, idx, color)
    image = add_legend_all_classes(image, classes)
    return image


def draw_segmentation(image: np.ndarray, detection: Mask) -> List[np.ndarray]:
    source_image = image.copy()
    images = []
    classes = [
        class_name
        for class_number, class_name in enumerate(detection.classes)
        if class_number in detection.mask["classes"]
    ]
    for class_number, rle_mask in enumerate(detection.mask["binary"]):
        one_class_image = source_image.copy()
        color = RGBA_COLORS[class_number]
        decoded_mask = decode_coco_rle(rle_mask)
        idx = decoded_mask == 1
        one_class_image = add_mask(one_class_image, idx, color)
        one_class_image = add_legend(one_class_image, classes, color, class_number)
        images.append(one_class_image)
    images.append(add_legend(source_image, classes, BACKGROUND_COLOR, -1))
    return images


def draw_classification_one_frame(image: np.ndarray, labels: List[Label]) -> np.ndarray:
    labels = [label for label in labels if label.score > 0.01]
    img_h, img_w, _ = image.shape
    scale = min([img_w, img_h]) / NORM_HEIGHT
    box_offset_x = box_offset_y = int(scale * BOX_CORNER_OFFSET)
    box_h = int(scale * PROB_BOX_HEIGHT)
    text_offset_x = int(scale * PROB_COMMENT_OFFSET_X)
    text_size = int(scale * COMMENT_TEXT_SIZE)
    box_w = max(
        [int(scale * PROB_BOX_WIDTH)]
        + [
            get_text_width(
                image,
                label.class_name.capitalize(),
                text_size,
                MONTSERATT_REGULAR_TTF_PATH,
            )
            + text_offset_x * 2
            for label in labels
        ],
    )
    for label_number, label in enumerate(labels):
        delimiter = label_number != len(labels) - 1
        box_x1 = img_w - box_offset_x - box_w
        box_y1 = box_offset_y + label_number * box_h
        box_x2 = box_x1 + box_w
        box_y2 = box_y1 + box_h
        coords = [box_x1, box_y1, box_x2, box_y2]
        image = add_probability_box(
            image,
            "{}%".format(int(label.score * 100)),
            label.class_name.capitalize(),
            coords,
            delimiter,
        )
    return image


def draw_classification(image: np.ndarray, labels: List[Label]) -> List[np.ndarray]:
    source_image = image.copy()
    return [draw_classification_one_frame(image, labels), source_image]


def draw_text_detections_one_frame(
    image: np.ndarray, detections: List[TextPolygon],
) -> np.ndarray:
    for polygon in detections:
        image = add_poly(image, polygon, LAVANDER_COLOR, LINE_THINKNESS)
        if len(polygon.text.strip()):
            image = add_info(
                image,
                [polygon.points[0].x, polygon.points[0].y],
                LAVANDER_COLOR,
                polygon.text,
                BLACK_TEXT_COLOR,
            )
    return image


def draw_text_detections(
    image: np.ndarray, detections: List[TextPolygon],
) -> List[np.ndarray]:
    source_image = image.copy()
    return [draw_text_detections_one_frame(image, detections), source_image]


def draw_landmarks_one_frame(
    image: np.ndarray, detections: List[Landmarks], draw_bbox: bool = True,
) -> np.ndarray:
    for detection in detections:
        for keypoint_number, keypoint in enumerate(detection.keypoints):
            image = add_point(image, keypoint, RGBA_COLORS[keypoint_number])
        if draw_bbox:
            image = add_bbox(
                image,
                (
                    detection.bbox.x1,
                    detection.bbox.y1,
                    detection.bbox.x2,
                    detection.bbox.y2,
                ),
                LAVANDER_COLOR,
            )
    return image


def draw_landmarks(
    image: np.ndarray, detections: List[Landmarks], draw_bbox: bool = True,
) -> List[np.ndarray]:
    source_image = image.copy()
    return [draw_landmarks_one_frame(image, detections, draw_bbox), source_image]


def draw_keypoints_one_frame(image: np.ndarray, detections: List[Pose]) -> np.ndarray:
    for detection in detections:
        image = add_pose(image, detection, LAVANDER_COLOR)
    return image


def draw_keypoints(image: np.ndarray, detections: List[Pose]) -> List[np.ndarray]:
    source_image = image.copy()
    images = []
    for detection in detections:
        images.append(draw_keypoints_one_frame(image.copy(), [detection]))
    images.append(draw_keypoints_one_frame(image, detections))
    images.append(source_image)
    return images


def draw_mesh_one_frame(
    image: np.ndarray, detections: List[Landmarks], draw_bbox: bool = True,
) -> np.ndarray:
    for detection in detections:
        for keypoint in detection.keypoints:
            image = add_point(image, keypoint, DARK_PINK_COLOR, MESH_POINT_SIZE)
        if draw_bbox:
            image = add_bbox(
                image,
                (
                    detection.bbox.x1,
                    detection.bbox.y1,
                    detection.bbox.x2,
                    detection.bbox.y2,
                ),
                LAVANDER_COLOR,
            )
    return image


def draw_mesh(
    image: np.ndarray, detections: List[Landmarks], draw_bbox: bool = True,
) -> List[np.ndarray]:
    source_image = image.copy()
    return [draw_mesh_one_frame(image, detections, draw_bbox), source_image]


def draw_age_gender_recognition_one_frame(
    image: np.ndarray, detections: List[AgeGenderLabel],
) -> np.ndarray:
    for detection in detections:
        coords = max(detection.bbox.x1, 0), max(detection.bbox.y1, 0)
        gender = max(detection.genders, key=lambda x: x.score).class_name
        text = "{}, {} y.o.".format(
            "MEN" if gender == "male" else "WOMAN", detection.age,
        )
        image = add_info(image, coords, BACKGROUND_COLOR, text, WHITE_TEXT_COLOR)
    return image


def draw_age_gender_recognition(
    image: np.ndarray, detections: List[AgeGenderLabel],
) -> List[np.ndarray]:
    source_image = image.copy()
    images = []
    for detection in detections:
        images.append(draw_age_gender_recognition_one_frame(image.copy(), [detection]))
    images.append(draw_age_gender_recognition_one_frame(image.copy(), detections))
    images.append(source_image)
    return images


def draw_emotion_recognition_one_frame(
    image: np.ndarray, detections: List[EmotionLabel],
) -> np.ndarray:
    scale = min(image.shape[:2]) / NORM_HEIGHT
    box_h = int(scale * PROB_BOX_HEIGHT)
    for label in detections:
        emotion = max(label.emotions, key=lambda x: x.score)
        x = max(label.bbox.x1, 0)
        y = max(label.bbox.y1 - box_h, 0)
        box_w = max(
            int(scale * PROB_BOX_WIDTH),
            get_text_width(
                image,
                emotion.class_name.capitalize(),
                int(scale * COMMENT_TEXT_SIZE),
                MONTSERATT_REGULAR_TTF_PATH,
            )
            + int(scale * PROB_COMMENT_OFFSET_X) * 2,
        )
        image = add_probability_box(
            image,
            "{}%".format(int(emotion.score * 100)),
            emotion.class_name.capitalize(),
            [x, y, x + box_w, y + box_h],
            False,
        )
    return image


def draw_emotion_recognition(
    image: np.ndarray, detections: List[EmotionLabel],
) -> List[np.ndarray]:
    images = []
    source_image = image.copy()
    for detection in detections:
        images.append(draw_emotion_recognition_one_frame(image.copy(), [detection]))
    images.append(draw_emotion_recognition_one_frame(image, detections))
    images.append(source_image)
    return images


def draw_tracks_one_frame(image: np.ndarray, detection: VideoFrame) -> np.ndarray:
    for box in detection.boxes:
        image = add_bbox(image, [box.x1, box.y1, box.x2, box.y2], LAVANDER_COLOR)
        image = add_info(
            image,
            (box.x1, box.y1),
            LAVANDER_COLOR,
            f"id:{box.track_number}",
            BLACK_TEXT_COLOR,
        )
    return image


def draw_tracks(
    video: Generator, frames: List[VideoFrame], save_path: str, fps: int,
) -> None:
    outputdict = FFMPEG_OUTPUT_DICT.copy()
    outputdict["-r"] = str(fps)
    writer = skvideo.io.FFmpegWriter(save_path, outputdict=outputdict)
    for frame_number, frame in enumerate(video):
        if frame_number >= len(frames):
            continue
        anno = frames[frame_number]
        frame = draw_tracks_one_frame(frame, anno)
        writer.writeFrame(frame)
    writer.close()


def draw_countable_tracks_one_frame(
    image: np.ndarray, detection: CountableVideoFrame,
) -> np.ndarray:
    for box in detection.boxes:
        image = add_bbox(image, [box.x1, box.y1, box.x2, box.y2], LAVANDER_COLOR)
        image = add_info(
            image,
            (box.x1, box.y1),
            LAVANDER_COLOR,
            f"id:{box.track_number}",
            BLACK_TEXT_COLOR,
        )
    img_h, img_w = image.shape[:2]
    scale = min(img_w, img_h) / NORM_HEIGHT
    box_offset_x = box_offset_y = int(scale * BOX_CORNER_OFFSET)
    box_h = box_w = int(scale * PROB_BOX_WIDTH)
    if detection.people_in is not None:
        image = add_probability_box(
            image,
            str(detection.people_in),
            "Came",
            [
                img_w - box_offset_x - box_w * 2,
                box_offset_y,
                img_w - box_offset_x - box_w,
                box_offset_y + box_h,
            ],
            False,
            CORALL_COLOR,
        )
    if detection.people_out is not None:
        image = add_probability_box(
            image,
            str(detection.people_out),
            "Left",
            [
                img_w - box_offset_x - box_w,
                box_offset_y,
                img_w - box_offset_x,
                box_offset_y + box_h,
            ],
            False,
            BACKGROUND_COLOR,
        )
    return image


def draw_countable_tracks(
    video: Generator, frames: List[CountableVideoFrame], save_path: str, fps: int,
) -> None:
    outputdict = FFMPEG_OUTPUT_DICT.copy()
    outputdict["-r"] = str(fps)
    writer = skvideo.io.FFmpegWriter(save_path, outputdict=outputdict)
    for frame_number, frame in enumerate(video):
        if frame_number >= len(frames):
            continue
        anno = frames[frame_number]
        frame = draw_countable_tracks_one_frame(frame, anno)
        writer.writeFrame(frame)
    writer.close()


def create_gif(images: List[np.ndarray], save_path: str, fps: int = 1) -> None:
    imageio.mimsave(save_path, images, format="GIF-FI", fps=fps, quantizer="nq")


def create_image(image: np.ndarray, save_path: str) -> None:
    cv2.imwrite(save_path, image)


def draw_background_removal(image: np.ndarray, detection: Mask) -> np.ndarray:
    foreground_mask_id = detection.classes.index("foreground")
    if len(detection.mask["binary"]) > foreground_mask_id:
        decoded_mask = decode_coco_rle(detection.mask["binary"][foreground_mask_id])
        idx = decoded_mask == 0
        image[idx] = (255, 255, 255)
        alfa_channel = decoded_mask.astype(image.dtype)[:, :, np.newaxis] * 255
    else:
        alfa_channel = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype)

    image = np.concatenate((image, alfa_channel), axis=2)
    return image
