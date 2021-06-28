# -*- coding: utf-8 -*-

import codecs
import os.path

from setuptools import setup


def get_version(rel_path):
    with codecs.open(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), rel_path), "r",
    ) as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


packages = ["modelplace_api"]

package_data = {"": ["*", "text_styles/*"]}
install_requires = [
    "pydantic>=1.5.1,<1.9.0",
    "loguru>=0.5.1",
    "numpy>=1.16.4",
]

extras_require = {
    "vis": [
        "Pillow>=7.1.2",
        "opencv-python>=4.2.0.34,<5.0",
        "imageio==2.9.0",
        "sk-video==1.1.10",
        "pycocotools==2.0.2",
    ],
    "vis-windows": [
        "Pillow>=7.1.2",
        "opencv-python>=4.2.0.34,<5.0",
        "imageio==2.9.0",
        "sk-video==1.1.10",
    ],
}

setup_kwargs = {
    "name": "modelplace-api",
    "version": get_version("modelplace_api/__init__.py"),
    "description": "",
    "long_description": None,
    "author": "OpenCV.AI",
    "author_email": "modelplace@opencv.ai",
    "maintainer": "OpenCV.AI",
    "maintainer_email": "modelplace@opencv.ai",
    "url": None,
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.6,<4.0",
    "extras_require": extras_require,
}

setup(**setup_kwargs)
