from matplotlib import colors
from flask import send_file # TODO
import numpy as np
import tifffile
import zarr
from PIL import Image
import base64
import openai
import json
import sys
import os
import io


def encode_image_url(array):
    encoding = "png"
    f = io.BytesIO()
    Image.fromarray(array, "RGB").save(f, format="PNG")
    encoded_image = base64.b64encode(f.getvalue()).decode("utf-8")
    return "data:image/{};base64,{}".format(encoding, encoded_image)


def make_questions_from_image(client, image_url, text, model):

    ROLE_DESCRIPTION = [
        "you are a expert pathology teacher at a prestigious university, your task is to generate exam question for your students based on the given H&E image and some notes about it, your exam questions are the best at showing the full range of the student knwoledge"
    ]

    OUTPUT_DESCRIPTION = [
        "provide no placeholder text",
        "output strictly in json format with the following template without any extra padding or visual formatting",
        "json output must follow this template {\"group1\":list of questions, \"group2\": list of questions. \"group3\": list of questions} where each question is formatted like this: {\"question\":question, \"a\":option_a, \"b\":option_b, \"c\":option_c, \"d\":option_d, \"answer\":answer}...},\"group2\":{...},\"group3\":{...}} where answer is one of \"a\", \"b\", \"c\", or \"d\" and the correct answer",
        "json output must not be formatted in any way, no newlines or tabs",
        "all questions must be nested and grouped under a single group",
        "provide 3 groups of 5 questions, the first group should contain 5 straight forward questions about the general image, the second group should have 5 questions about fine grain specifics and the third group should contain 5 very specialized questions about the image",
        "questions must be multiple choice of 4 options and exactly one option is the correct one",
        "the generated questions must strictly be about the given image and no extra information is needed to answer correctly"
    ]

    ROLE_DESCRIPTION = [
        {"role": 'system', "content": x} for x in ROLE_DESCRIPTION
    ]
    OUTPUT_DESCRIPTION = [
        {"role": 'system', "content": x} for x in OUTPUT_DESCRIPTION
    ]

    client_msg = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
        }
    ]

    if text is not None:
        client_msg += [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]

    role_msg = ROLE_DESCRIPTION + OUTPUT_DESCRIPTION

    response = client.chat.completions.create(
        model=model, messages=role_msg + client_msg
    )

    for res in response.choices:
        print(res.message.content)
        
    return response


class ZarrWrapper:

    def __init__(self, group, dim_list):

        self.group = group

        self.dim_list = dim_list

    def __getitem__(self, full_idx_list):
        """
        Access zarr groups as if in a standard dimension order
        Args:
            full_idx_list: level, x range, y range, z index, channel number, timestep
        """

        level = full_idx_list[0]
        idx_list = full_idx_list[1:]

        # Use dimensions from standard order
        tile = self.group[level].__getitem__(tuple([
            idx_list[["X", "Y", "Z", "C", "T"].index(key)]
            for key in self.dim_list
        ]))
        # Remove additional dimensions
        if tile.ndim > 2:
            tile = np.squeeze(
                tile, axis=tuple(range(2, tile.ndim))
            )

        return tile


def composite_channel(target, image, color, range_min, range_max):
    """Render _image_ in pseudocolor and composite into _target_
    Args:
        target: Numpy float32 array containing composition target image
        image: Numpy array of image to render and composite
        color: Color as r, g, b float array, 0-1
        range_min: Threshold range minimum (in terms of image pixel values)
        range_max: Threshold range maximum (in terms of image pixel values)
    """
    if range_min == range_max:
        return
    f_image = (image.astype("float32") - range_min) / (range_max - range_min)
    f_image = f_image.clip(0, 1, out=f_image)
    for i, component in enumerate(color):
        target[:, :, i] += f_image * component


def viewport_to_image(
    level_image_height, pan
):
    return (
        pan[0] * level_image_height,
        pan[1] * level_image_height
    )


def nearest_power_of_two( reference, x0, x1, y0, y1 ):
    return np.round(np.log2(
        max(x1-x0, y1-y0) / reference
    )).astype(int)


def waypoint_to_image_rect(
    waypoint, level_image_height, level_image_width
):
    pan = waypoint["Overlay"]["Pan"]
    min_x, max_x = sorted(pan[:2])
    min_y, max_y = sorted(pan[2:])
    (x0, y0) = viewport_to_image(
        level_image_height, [ min_x, min_y ]
    )
    (x1, y1) = viewport_to_image(
        level_image_height, [ max_x, max_y ] 
    )
    # Constrain coordinates
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = max(x1, x0+1)
    y1 = max(y1, y0+1)
    x1 = min(x1, level_image_width)
    y1 = min(y1, level_image_height)
    return tuple( int(v) for v in ( x0, x1, y0, y1 ) )


def composite_image(
    waypoint, image_path, selected_channel_settings
):
    target = None
    best_size = 1024
    series = tifffile.TiffFile(image_path).series
    group = zarr.open(series[0].aszarr())
    dim_list = [
        {"I": "C", "S": "C"}.get(d, d)
        for d in series[0].get_axes()
    ]
    # Treat non-pyramids as groups of one array
    if isinstance(group, zarr.core.Array):
        root = zarr.group()
        root[0] = group
        group = root
    wrapper = ZarrWrapper(group, dim_list)
    # Calculate full-scale coordinates
    ( x0, x1, y0, y1 ) = waypoint_to_image_rect(
        waypoint, group[0].shape[dim_list.index("Y")],
        group[0].shape[dim_list.index("X")]
    )
    best_level = nearest_power_of_two(
        best_size, x0, x1, y0, y1 
    )
    level = min(best_level, len(group)-1)
    ( x0, x1, y0, y1 ) = waypoint_to_image_rect(
        waypoint, group[level].shape[dim_list.index("Y")],
        group[level].shape[dim_list.index("X")]
    )
    sample = 2**nearest_power_of_two(
        best_size, x0, x1, y0, y1 
    )

    for settings in selected_channel_settings:
        ( channel_id, color, range_min, range_max ) = (
            settings[k] for k in ("id", "color", "min", "max")
        )
        image = wrapper[
            level, x0:x1:sample, y0:y1:sample, 0, channel_id, 0
        ]
        if target is None:
            target = np.zeros(
                (image.shape[0], image.shape[1], 3),
                np.float32
            )

        iinfo = np.iinfo(image.dtype)
        composite_channel(
            target, image, colors.to_rgb(color),
            float(range_min) * iinfo.max,
            float(range_max) * iinfo.max
        )

    np.clip(target, 0, 1, out=target)
    return np.rint(target * 255).astype(np.uint8)


def to_quiz_waypoint(
    client, waypoint, image_path,
    selected_channel_settings, model
):
    array = composite_image(
        waypoint, image_path, selected_channel_settings
    )
    text = waypoint["Description"]
    image_url = encode_image_url(array)
    response = make_questions_from_image(
        client, image_url, text, model
    )
    try:
        choice = response.choices[0]
        output = json.loads(choice.message.content)["group1"][0]
        question = output["question"]
        answer = output["answer"]
        a = output["a"]
        b = output["b"]
        c = output["c"]
        d = output["d"]
        correct = (
            f': "{output[answer]}"'
        ) if answer in output else ''
        description = f'''## Question
{question}

## Options

A. {a}  
B. {b}  
C. {c}  
D. {d}  

## Correct Answer

{answer.upper()}{correct}
'''
        return {
            **waypoint, "Description": description
        }
    except IndexError as e:
        print(e, file=sys.stderr)


def to_quiz_stories(
    client, exhibit, image_path,
    selected_channel_settings, model
):
    for in_story in exhibit["Stories"]:
        in_waypoints = in_story["Waypoints"]
        out_waypoints = []
        for waypoint in in_waypoints:
            out_waypoints.append(to_quiz_waypoint(
                client, waypoint, image_path,
                selected_channel_settings, model
            ))
        yield {
            **in_story, "Waypoints": list(out_waypoints)
        }


def update_exhibit_with_llm(
    exhibit, image_path, selected_channel_settings, model
):
    client = openai.AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-05-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    stories = to_quiz_stories(
        client, exhibit, image_path,
        selected_channel_settings, model
    )
    return {
        **exhibit, "Stories": list(stories)
    }


def main(
    in_text, in_image_path, roi_coordinates, selected_channel_settings
):
    in_exhibit = {
        "Stories": [{
            "Waypoints": [{
                "Overlay": {
                    "Pan": roi_coordinates,
                },
                "Description": in_text
            }]
        }]
    }
    model = 'gpt-4o-mini'
    out_exhibit = update_exhibit_with_llm(
        in_exhibit, in_image_path,
        selected_channel_settings, model
    )
    return out_exhibit["Stories"][0]["Waypoints"][0]["Description"]
