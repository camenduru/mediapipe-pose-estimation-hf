#!/usr/bin/env python

from __future__ import annotations

import pathlib

import gradio as gr
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

TITLE = 'MediaPipe Human Pose Estimation'
DESCRIPTION = 'https://google.github.io/mediapipe/'


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
            bg_color = (0, 255, 0)  # type: ignore
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


model_complexities = list(range(3))
background_colors = ['white', 'black', 'green']

image_paths = sorted(pathlib.Path('images').rglob('*.jpg'))
examples = [[path, model_complexities[1], True, 0.5, background_colors[0]]
            for path in image_paths]

gr.Interface(
    fn=run,
    inputs=[
        gr.Image(label='Input', type='numpy'),
        gr.Radio(label='Model Complexity',
                 choices=model_complexities,
                 type='index',
                 value=model_complexities[1]),
        gr.Checkbox(label='Enable Segmentation', value=True),
        gr.Slider(label='Minimum Detection Confidence',
                  minimum=0,
                  maximum=1,
                  step=0.05,
                  value=0.5),
        gr.Radio(label='Background Color',
                 choices=background_colors,
                 type='value',
                 value=background_colors[0]),
    ],
    outputs=gr.Image(label='Output', height=500),
    examples=examples,
    title=TITLE,
    description=DESCRIPTION,
).queue().launch()
