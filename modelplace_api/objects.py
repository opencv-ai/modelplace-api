import enum
from typing import List

import pydantic


class Point(pydantic.BaseModel):
    x: int
    y: int


class TextPolygon(pydantic.BaseModel):
    points: List[Point] = []
    text: str = ""


class BBox(pydantic.BaseModel):
    x1: float = 0
    y1: float = 0
    x2: float = 1
    y2: float = 1
    score: float = 0
    class_name: str = 0


class TrackBBox(pydantic.BaseModel):
    x1: float = 0
    y1: float = 0
    x2: float = 1
    y2: float = 1
    score: float = 0
    class_name: str = 0
    track_number: int = -1


class COCOBBox(pydantic.BaseModel):
    x1: float = 0
    y1: float = 0
    x2: float = 1
    y2: float = 1
    score: float = 0
    class_name: str = 0
    area: float = 0
    is_crowd: int = 0


class Mask(pydantic.BaseModel):
    mask: dict
    classes: list


class Joint(pydantic.BaseModel):
    x: int = 0
    y: int = 0
    class_name: str = 0
    score: float = 1


class Link(pydantic.BaseModel):
    joint_a: Joint
    joint_b: Joint


class Pose(pydantic.BaseModel):
    score: float = 0
    links: List[Link] = []
    skeleton_parts: list = []


class COCOPose(pydantic.BaseModel):
    score: float = 0
    links: List[Link] = []
    skeleton_parts: list = []
    # GT PARTS
    bbox: COCOBBox = COCOBBox()


class Label(pydantic.BaseModel):
    score: float = 0
    class_name: str = 0


class AgeGenderLabel(pydantic.BaseModel):
    bbox: BBox = BBox()
    age: int = 0
    gender: List[Label] = []


class EmotionLabel(pydantic.BaseModel):
    bbox: BBox = BBox()
    emotion: List[Label] = []


class FacialLandmarks(pydantic.BaseModel):
    bbox: BBox = BBox()
    keypoints: List[Point] = []


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
    facial_landmark_detection = enum.auto()
    age_gender_recognition = enum.auto()
    emotion_recognition = enum.auto()
