from collections import defaultdict
from typing import Any, Generator, List, Union

from .colors import RGB_COLORS
from .objects import (
    AgeGenderLabel,
    BBox,
    CountableVideoFrame,
    EmotionLabel,
    FacialLandmarks,
    Label,
    Mask,
    Pose,
    TextPolygon,
    VideoFrame,
)

try:
    import cv2
    import imageio
    import numpy as np
    import skvideo.io
    from PIL.Image import Image
    from pycocotools import mask
except ImportError:
    raise ImportError(
        "Some dependencies is invalid. "
        "Please install this package with extra requiements: pip install modelplace-api[vis]",
    )


FFMPEG_OUTPUT_DICT = {
    "-vcodec": "libx264",
    "-vf": "format=yuv420p",
    "-movflags": "+faststart",
}


def draw_line(
    img: np.ndarray,
    pt1: tuple,
    pt2: tuple,
    color: tuple,
    thickness: int = 1,
    style: str = "dotted",
    gap: int = 7,
) -> None:
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        p = (x, y)
        pts.append(p)

    if style == "dotted":
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        e = pts[0]
        i = 0
        for p in pts:
            s, e = e, p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def draw_text_label(
    image: np.ndarray,
    detection: AgeGenderLabel,
    text: str,
    font_scale: float = 0.6,
    thickness: int = 1,
    bg_color: tuple = (128, 128, 128),
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_TRIPLEX
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=thickness,
    )[0]
    text_offset_x = detection.bbox.x1
    text_offset_y = detection.bbox.y1 - 10
    box_coords = (
        (int(text_offset_x), int(text_offset_y)),
        (int(text_offset_x + text_width + 2), int(text_offset_y - text_height - 2)),
    )
    cv2.rectangle(image, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(
        image,
        text,
        (int(text_offset_x), int(text_offset_y)),
        font,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=thickness,
    )
    return image


def draw_text(
    img: np.ndarray,
    text: str,
    origin: tuple,
    thickness: int = 1,
    bg_color: tuple = (128, 128, 128),
    font_scale: float = 0.5,
) -> np.ndarray:
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    baseline += thickness
    text_org = np.array((origin[0], origin[1] - text_size[1]))
    cv2.rectangle(
        img,
        tuple((text_org + (0, baseline)).astype(int)),
        tuple((text_org + (text_size[0], -text_size[1])).astype(int)),
        bg_color,
        -1,
    )

    cv2.putText(
        img,
        text,
        tuple((text_org + (0, baseline / 2)).astype(int)),
        font_face,
        font_scale,
        (0, 0, 0),
        thickness,
        8,
    )

    return img


def draw_poly(
    img: np.ndarray, pts: list, color: tuple, thickness: int = 1, style: str = "dotted",
) -> None:
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s, e = e, p
        draw_line(img, s, e, color, thickness, style, gap=1)


def draw_legend(
    image: np.ndarray,
    class_map: dict,
    thickness: int = 3,
    cell_height: int = 20,
    height: int = 10,
    weight: int = 10,
) -> np.ndarray:
    bg_color = (255, 255, 255)
    x0 = 0
    y0 = 0
    legend = np.zeros((image.shape[0], 200, 3), dtype=image.dtype)
    legend.fill(255)
    cv2.rectangle(
        legend,
        (y0, x0),
        (x0 + 200, y0 + cell_height * (len(class_map.keys()) + 3) + 5),
        bg_color,
        -1,
    )
    draw_text(legend, "Labels:", (weight, height + 15), bg_color=bg_color)
    height += cell_height
    for class_name, class_color in class_map.items():
        cv2.line(
            legend, (weight + 2, height), (weight + 40, height), class_color, thickness,
        )
        draw_text(legend, class_name, (weight + 45, height + 15), bg_color=bg_color)
        height += cell_height
    return np.concatenate([image, legend], axis=1)


def draw_classification_legend(
    image: np.ndarray,
    class_map: dict,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    width_offset: int = 10,
    height_offset: int = 15,
    cell_height: int = 40,
    height: int = 10,
    weight: int = 10,
) -> np.ndarray:
    bg_color = (255, 255, 255)
    x0 = 0
    y0 = 0
    legend = np.zeros((image.shape[0], 300, 3), dtype=image.dtype)
    legend.fill(255)
    cv2.rectangle(
        legend,
        (y0, x0),
        (x0 + 300, y0 + cell_height * (len(class_map.keys()) + 3) + 5),
        bg_color,
        -1,
    )
    draw_text(
        legend,
        "Labels:",
        (weight, height + height_offset),
        bg_color=bg_color,
        font_scale=font_scale,
        thickness=font_thickness,
    )
    height += cell_height
    for class_name, class_score in class_map.items():
        draw_text(
            legend,
            f"{class_name}: {class_score}",
            (weight + width_offset, height),
            bg_color=bg_color,
            font_scale=font_scale,
            thickness=font_thickness,
        )
        height += cell_height
    return np.concatenate([image, legend], axis=1)


def draw_detection_result(
    image: Union[Image, np.ndarray],
    detections: List[BBox],
    confidence_threshold: float = 0,
    thickness: int = 2,
) -> List[np.ndarray]:
    image_with_boxes = np.ascontiguousarray(image)
    source_image = image_with_boxes.copy()
    images = []
    detections = list(filter(lambda x: x.score > confidence_threshold, detections))
    possible_labels = list(set([det.class_name for det in detections]))
    class_map = dict(
        [
            [possible_labels[class_number], RGB_COLORS[class_number][::-1]]
            for class_number in range(len(possible_labels))
        ],
    )

    images.append(source_image)
    classes = defaultdict(list)
    for det in detections:
        classes[det.class_name].append(det)
    for class_name, class_detections in classes.items():
        one_class_image = source_image.copy()
        for detection in class_detections:
            label_id = possible_labels.index(detection.class_name)
            color = RGB_COLORS[label_id][::-1]
            image_with_boxes = cv2.rectangle(
                image_with_boxes,
                (int(detection.x1), int(detection.y1)),
                (int(detection.x2), int(detection.y2)),
                tuple(color),
                thickness=thickness,
            )
            one_class_image = cv2.rectangle(
                one_class_image,
                (int(detection.x1), int(detection.y1)),
                (int(detection.x2), int(detection.y2)),
                tuple(color),
                thickness=thickness,
            )
        images.append(one_class_image)
    images.append(image_with_boxes)
    return [draw_legend(image, class_map) for image in images]


def draw_segmentation_result(
    image: Union[Image, np.ndarray], detection: Mask,
) -> List[np.ndarray]:
    image_with_mask = np.ascontiguousarray(image)
    source_image = image_with_mask.copy()
    images = []
    rgb_mask = np.zeros_like(image).astype(np.uint8)
    class_map = dict(
        [
            [detection.classes[ids], RGB_COLORS[ids + 160][:3][::-1]]
            for ids in detection.mask["classes"]
        ],
    )
    images.append(source_image)
    for m, cl in zip(detection.mask["binary"], detection.mask["classes"]):
        m = mask.decode(m)

        one_class_mask = np.zeros_like(image).astype(np.uint8)
        color = class_map[detection.classes[cl]]

        rgb_mask[m == 1] = color
        one_class_mask[m == 1] = color

        images.append(cv2.addWeighted(image_with_mask, 0.5, one_class_mask, 0.5, 0))

    images.append(cv2.addWeighted(image_with_mask, 0.55, rgb_mask, 0.45, 0))
    return [draw_legend(image, class_map) for image in images]


def draw_pose_estimation_result(
    image: Union[Image, np.ndarray],
    detections: List[Pose],
    confidence_threshold: float,
    simple_labels: bool = False,
) -> List[np.ndarray]:
    image_with_skeletons = np.ascontiguousarray(image)
    source_img = image_with_skeletons.copy()
    images = [source_img]
    class_map = {}
    for detection in detections:
        if detection.score > confidence_threshold:
            one_pose_image = source_img.copy()
            for link in detection.links:
                if (
                    link.joint_b.x == link.joint_b.y == 0
                    or link.joint_a.x == link.joint_a.y == 0
                ):
                    continue

                cv2.line(
                    image_with_skeletons,
                    (int(link.joint_a.x), int(link.joint_a.y)),
                    (int(link.joint_b.x), int(link.joint_b.y)),
                    RGB_COLORS[-1][2::-1],
                    2,
                )
                cv2.line(
                    one_pose_image,
                    (int(link.joint_a.x), int(link.joint_a.y)),
                    (int(link.joint_b.x), int(link.joint_b.y)),
                    RGB_COLORS[-1][2::-1],
                    2,
                )

            joints = [
                joint
                for link in detection.links
                for joint in [link.joint_a, link.joint_b]
            ]
            unique_joints = []
            for i, a in enumerate(joints):
                if not any(a.class_name == b.class_name for b in joints[:i]):
                    class_map[a.class_name] = RGB_COLORS[len(unique_joints)][2::-1]
                    unique_joints.append(a)
            for i, joint in enumerate(unique_joints):
                if joint.x == joint.y == 0:
                    continue
                if not simple_labels:
                    cv2.circle(
                        image_with_skeletons,
                        (int(joint.x), int(joint.y)),
                        4,
                        RGB_COLORS[i][2::-1],
                        -1,
                    )
                    cv2.circle(
                        one_pose_image,
                        (int(joint.x), int(joint.y)),
                        4,
                        RGB_COLORS[i][2::-1],
                        -1,
                    )
                else:
                    cv2.circle(
                        image_with_skeletons,
                        (int(joint.x), int(joint.y)),
                        4,
                        RGB_COLORS[len(unique_joints)][2::-1],
                        -1,
                    )
                    cv2.circle(
                        one_pose_image,
                        (int(joint.x), int(joint.y)),
                        4,
                        RGB_COLORS[len(unique_joints)][2::-1],
                        -1,
                    )
            images.append(one_pose_image)
            if simple_labels:
                class_map = {"keypoint": RGB_COLORS[len(unique_joints)][2::-1]}

    images.append(image_with_skeletons)
    return [draw_legend(image, class_map) for image in images]


def draw_landmarks_result(
    image: Union[Image, np.ndarray],
    detections: List[FacialLandmarks],
    classes: List[str],
    mapping_classes_to_points: dict,
    confidence_threshold: float,
) -> List[np.ndarray]:
    image = np.ascontiguousarray(image)
    image_with_bbox = image.copy()
    images = []
    images.append(image)

    class_map = dict(
        [[classes[ids], RGB_COLORS[ids + 1][:3][::-1]] for ids in range(len(classes))],
    )
    for detection in detections:
        image_with_bbox = cv2.rectangle(
            image_with_bbox,
            (int(detection.bbox.x1), int(detection.bbox.y1)),
            (int(detection.bbox.x2), int(detection.bbox.y2)),
            RGB_COLORS[196],
            thickness=6,
        )
    images.append(image_with_bbox)
    image_with_landmarks = image_with_bbox.copy()
    for detection in detections:
        keypoints = [(int(item.x), int(item.y)) for item in detection.keypoints]
        for class_name, keypoints_idx in mapping_classes_to_points.items():
            color = class_map[class_name]
            for keypoint_idx in keypoints_idx:
                keypoint = keypoints[keypoint_idx]
                image_with_landmarks = cv2.circle(
                    image_with_landmarks, keypoint, radius=2, color=color, thickness=5,
                )
    images.append(image_with_landmarks)

    return [draw_legend(image, class_map) for image in images]


def draw_text_detections(
    image: Union[Image, np.ndarray], detections: List[TextPolygon],
) -> List[np.ndarray]:
    image = np.ascontiguousarray(image)
    color = RGB_COLORS[1][::-1]
    images = [image.copy()]
    for polygon in detections:
        draw_poly(
            image,
            [[point.x, point.y] for point in polygon.points],
            color,
            thickness=2,
            style="",
        )
    images.append(image)

    return [draw_legend(image, {"Text": color}) for image in images]


def draw_tracks(
    video: Any, frames: List[VideoFrame], save_path, color=RGB_COLORS[2],
) -> None:
    writer = skvideo.io.FFmpegWriter(save_path, outputdict=FFMPEG_OUTPUT_DICT)

    for frame_number, frame in enumerate(video):
        if frame_number >= len(frames):
            continue

        anno = frames[frame_number]
        for box in anno.boxes:
            cv2.rectangle(
                frame,
                (int(box.x1), int(box.y1)),
                (int(box.x2), int(box.y2)),
                tuple(color),
                2,
            )
            draw_text(
                frame,
                f"id:{box.track_number}",
                (int(box.x1), int(box.y1)),
                1,
                (255, 255, 255),
            )
        writer.writeFrame(frame)

    writer.close()


def draw_countable_tracks(
    video: Generator[np.ndarray, None, None],
    frames: List[CountableVideoFrame],
    save_path,
) -> None:
    writer = skvideo.io.FFmpegWriter(save_path, outputdict=FFMPEG_OUTPUT_DICT)

    for frame_number, frame in enumerate(video):
        if frame_number >= len(frames):
            continue
        anno = frames[frame_number]
        for box in anno.boxes:
            color = RGB_COLORS[box.track_number % len(RGB_COLORS)]
            cv2.rectangle(
                frame,
                (int(box.x1), int(box.y1)),
                (int(box.x2), int(box.y2)),
                tuple(color),
                2,
            )
            draw_text(
                frame,
                f"id:{box.track_number}",
                (int(box.x1), int(box.y1)),
                1,
                (255, 255, 255),
            )

        if anno.people_in is not None:
            draw_text(
                frame, f"IN: {anno.people_in}", (100, 100), 1, (0, 255, 0), 1,
            )

        if anno.people_out is not None:
            draw_text(
                frame, f"OUT: {anno.people_out}", (100, 200), 1, (255, 0, 0), 1,
            )
        writer.writeFrame(frame)

    writer.close()


def draw_classification_result(
    image: Union[Image, np.ndarray], detections: List[Label],
) -> List[np.ndarray]:
    image = np.ascontiguousarray(image)
    class_map = {label.class_name.capitalize(): label.score for label in detections}

    return [draw_classification_legend(image, class_map)]


def draw_age_gender_recognition_result(
    image: Union[Image, np.ndarray], detections: List[AgeGenderLabel],
) -> List[np.ndarray]:
    image_with_boxes = np.ascontiguousarray(image)
    source_image = image_with_boxes.copy()
    images = list()
    images.append(source_image)
    for detection in detections:
        image_with_boxes = cv2.rectangle(
            image_with_boxes,
            (int(detection.bbox.x1), int(detection.bbox.y1)),
            (int(detection.bbox.x2), int(detection.bbox.y2)),
            RGB_COLORS[196],
            thickness=8,
        )
        bg_color = RGB_COLORS[196]
        label_score, label_class_name = detection.gender[0]
        age = detection.age
        text = f"{label_class_name[1]}, Age: {age}"
        image_with_boxes = draw_text_label(
            image_with_boxes, detection, text, bg_color=bg_color,
        )
        images.append(image_with_boxes)
    return images


def draw_emotion_recognition_result(
    image: Union[Image, np.ndarray], detections: List[AgeGenderLabel],
) -> List[np.ndarray]:
    image_with_boxes = np.ascontiguousarray(image)
    source_image = image_with_boxes.copy()
    images = list()
    images.append(source_image)
    for detection in detections:
        image_with_boxes = cv2.rectangle(
            image_with_boxes,
            (int(detection.bbox.x1), int(detection.bbox.y1)),
            (int(detection.bbox.x2), int(detection.bbox.y2)),
            RGB_COLORS[196],
            thickness=8,
        )
        bg_color = RGB_COLORS[196]
        label_score, label_class_name = detection.emotion[0]
        image_with_boxes = draw_text_label(
            image_with_boxes, detection, label_class_name[1], bg_color=bg_color,
        )
        images.append(image_with_boxes)
    return images


def create_gif(images: List[np.ndarray], save_path: str, fps: int = 1) -> None:
    imageio.mimsave(save_path, images, format="GIF-FI", fps=fps, quantizer="nq")


classes_adas = [
    "Left Eye",
    "Right Eye",
    "Nose",
    "Mouth",
    "Left Eyebrow",
    "Right Eyebrow",
    "Face Contour",
]
mapping_classes_to_points_adas = {
    classes_adas[0]: [0, 1],
    classes_adas[1]: [2, 3],
    classes_adas[2]: [4, 5, 6, 7],
    classes_adas[3]: [8, 9, 10, 11],
    classes_adas[4]: [12, 13, 14],
    classes_adas[5]: [15, 16, 17],
    classes_adas[6]: [x for x in range(18, 35)],
}

classes_retail = [
    "Left Eye",
    "Right Eye",
    "Nose",
    "Left Lip Corner",
    "Right Lip Corner",
]
mapping_classes_to_points_retail = {
    classes_retail[0]: [0],
    classes_retail[1]: [1],
    classes_retail[2]: [2],
    classes_retail[3]: [3],
    classes_retail[4]: [4],
}
