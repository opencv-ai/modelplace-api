import enum
from typing import List

import pydantic


class Point(pydantic.BaseModel):
    x: int
    y: int


class TextPolygon(pydantic.BaseModel):
    points: List[Point]
    text: str = ""


class BBox(pydantic.BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_name: str


class TrackBBox(pydantic.BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_name: str
    track_number: int


class COCOBBox(pydantic.BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_name: str
    area: float
    is_crowd: int


class Mask(pydantic.BaseModel):
    mask: dict
    classes: list


class InstanceMask(pydantic.BaseModel):
    detections: List[BBox]
    masks: List[Mask]


class Joint(pydantic.BaseModel):
    x: int
    y: int
    class_name: str
    score: float


class Link(pydantic.BaseModel):
    joint_a: Joint
    joint_b: Joint


class Pose(pydantic.BaseModel):
    score: float
    links: List[Link]
    skeleton_parts: List


class COCOPose(pydantic.BaseModel):
    score: float
    links: List[Link]
    skeleton_parts: List
    # GT PARTS
    bbox: COCOBBox


class COCOInstanceMask(pydantic.BaseModel):
    detections: List[COCOBBox]
    masks: List[Mask]


class Label(pydantic.BaseModel):
    score: float
    class_name: str


class AgeGenderLabel(pydantic.BaseModel):
    bbox: BBox
    age: int
    genders: List[Label]


class EmotionLabel(pydantic.BaseModel):
    bbox: BBox
    emotions: List[Label]


class Landmarks(pydantic.BaseModel):
    bbox: BBox
    keypoints: List[Point]


class VideoFrame(pydantic.BaseModel):
    number: int
    boxes: List[TrackBBox]


class CountableVideoFrame(VideoFrame):
    people_in: int
    people_out: int


class Device(enum.Enum):
    gpu = enum.auto()
    cpu = enum.auto()


@enum.unique
class TaskType(enum.Enum):
    detection = enum.auto()
    segmentation = enum.auto()
    pose_estimation = enum.auto()
    tracking = enum.auto()
    text_detection = enum.auto()
    unknown = enum.auto()
    people_counting = enum.auto()
    classification = enum.auto()
    landmark_detection = enum.auto()
    age_gender_recognition = enum.auto()
    emotion_recognition = enum.auto()
    mesh_detection = enum.auto()
    background_removal = enum.auto()
    instance_segmentation = enum.auto()
