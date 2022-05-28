#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import tarfile

if os.environ.get('SYSTEM') == 'spaces':
    subprocess.call('pip uninstall -y opencv-python'.split())
    subprocess.call('pip uninstall -y opencv-python-headless'.split())
    subprocess.call('pip install opencv-python-headless==4.5.5.64'.split())

import gradio as gr
import huggingface_hub
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

TITLE = 'MediaPipe Human Pose Estimation'
DESCRIPTION = 'https://google.github.io/mediapipe/'
ARTICLE = '<center><img src="https://visitor-badge.glitch.me/badge?page_id=hysts.mediapipe-pose-estimation" alt="visitor badge"/></center>'

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


def load_sample_images() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        image_dir.mkdir()
        dataset_repo = 'hysts/input-images'
        filenames = ['002.tar']
        for name in filenames:
            path = huggingface_hub.hf_hub_download(dataset_repo,
                                                   name,
                                                   repo_type='dataset',
                                                   use_auth_token=TOKEN)
            with tarfile.open(path) as f:
                f.extractall(image_dir.as_posix())
    return sorted(image_dir.rglob('*.jpg'))


def run(image: np.ndarray, model_complexity: int, enable_segmentation: bool,
        min_detection_confidence: float, background_color: str) -> np.ndarray:
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence) as pose:
        results = pose.process(image)

    res = image[:, :, ::-1].copy()
    if enable_segmentation:
        if background_color == 'white':
            bg_color = 255
        elif background_color == 'black':
            bg_color = 0
        elif background_color == 'green':
            bg_color = (0, 255, 0)
        else:
            raise ValueError

        if results.segmentation_mask is not None:
            res[results.segmentation_mask <= 0.1] = bg_color
        else:
            res[:] = bg_color

    mp_drawing.draw_landmarks(res,
                              results.pose_landmarks,
                              mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.
                              get_default_pose_landmarks_style())

    return res[:, :, ::-1]


def main():
    args = parse_args()

    model_complexities = list(range(3))
    background_colors = ['white', 'black', 'green']

    image_paths = load_sample_images()
    examples = [[
        path.as_posix(), model_complexities[1], True, 0.5, background_colors[0]
    ] for path in image_paths]

    gr.Interface(
        run,
        [
            gr.inputs.Image(type='numpy', label='Input'),
            gr.inputs.Radio(model_complexities,
                            type='index',
                            default=model_complexities[1],
                            label='Model Complexity'),
            gr.inputs.Checkbox(default=True, label='Enable Segmentation'),
            gr.inputs.Slider(0,
                             1,
                             step=0.05,
                             default=0.5,
                             label='Minimum Detection Confidence'),
            gr.inputs.Radio(background_colors,
                            type='value',
                            default=background_colors[0],
                            label='Background Color'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
