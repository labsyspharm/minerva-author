from matplotlib import colors
from pathlib import Path
import numpy as np
import zarr
from PIL import Image
import base64
import openai
import json
import sys
import os
import io


def return_image_opener(path):
    opener = return_opener(path, "image_openers")
    success = cache_image_opener(path, opener)
    return (not success, opener)


def encode_image_url(array):
    encoding = "png"
    f = io.BytesIO()
    Image.fromarray(array, "RGB").save(f, format="PNG")
    encoded_image = base64.b64encode(f.getvalue()).decode("utf-8")
    return "data:image/{};base64,{}".format(encoding, encoded_image)


def read_textbook_chapter(chapter):
    ch_path = f'robbins-cotran-ch{chapter}.md'
    with open(Path('src/robbins-cotran') / ch_path) as rf:
        return rf.read()


def make_questions_from_image(client, image_url, text, model):

    textbook_title = "Robbins & Cotran Pathologic Basis of Disease"
    selected_chapters = {
#        '7': 'Neoplasia',
#        '17': 'The Gastrointestinal Tract'
    }
    ROLE_DESCRIPTION = [
        "You are a expert pathology teacher at a prestigious university, your task is to generate exam questions for your students based on the given image and some notes about it, your exam questions are the best at showing the full range of the student knowledge.",
        #f"Your students already have studied the following chapters from \"{textbook_title}\" on "
        #+ " and ".join(str(c) for c in selected_chapters.values()) + "."
        #] + [
        #f"{chapter_title}\n\n{read_textbook_chapter(chapter)}" for
        #chapter, chapter_title in selected_chapters.items()
    ] + [
        f"Your exam questions must be answerable from the provided image.",
        f"Your exam questions must about the provided image.",
        f"Your exam questions must be unambiguous."
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


def select_image(waypoint, opener):
    best_size = 1024
    group = opener.group 
    dim_list = opener.wrapper.dim_list 
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
    level = max(0, min(best_level, len(group)-1))
    box = waypoint_to_image_rect(
        waypoint, group[level].shape[dim_list.index("Y")],
        group[level].shape[dim_list.index("X")]
    )
    sample = 2**max(
        0, nearest_power_of_two(best_size, *box)
    )
    return wrapper, level, sample, box


def yield_histograms(
    waypoint, opener, selected_channel_settings
):
    wrapper, level, sample, box = select_image(waypoint, opener)
    x0, x1, y0, y1 = box

    for settings in selected_channel_settings:
        ( channel_id, color, range_min, range_max ) = (
            settings[k] for k in ("id", "color", "min", "max")
        )
        image = wrapper[
            level, x0:x1:sample, y0:y1:sample, 0, channel_id, 0
        ]
        bins = np.logspace(1, 16, 750)
        print(bins)
        yield (
            channel_id, list(np.histogram(image, bins=bins, density=True)[0]) 
        )


def composite_image(
    waypoint, opener, selected_channel_settings
):
    wrapper, level, sample, box = select_image(waypoint, opener)
    x0, x1, y0, y1 = box
    target = None

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
    client, waypoint, opener,
    selected_channel_settings,
    make_quiz, model
):
    histograms = dict(yield_histograms(
        waypoint, opener, selected_channel_settings
    ))
    array = composite_image(
        waypoint, opener, selected_channel_settings
    )
    text = waypoint["Description"]
    image_url = encode_image_url(array)
    image_url_small = encode_image_url(array[::2,::2])
    # TODO
#    return {
#        "image": image_url,
#        "histograms": histograms 
#    }
    # TODO
    if not make_quiz:
        return {
            "image": image_url_small
        }
    response = make_questions_from_image(
        client, image_url, text, model
    )
    choice = response.choices[0]
    quiz_groups = {}
    # Try LLM up to thrice 
    for n in range(1, 4):
        try: 
            quiz_groups = json.loads(choice.message.content)
            break
        except json.decoder.JSONDecodeError as e:
            print(f"Failed LLM attempt #{n} due to JSON issue")
            pass
    return {
        "image": image_url_small,
        "groups": quiz_groups 
        #"groups": {"group1":[{"a":"Hematoxylin and Eosin","answer":"a","b":"Gram stain","c":"Masson's trichrome","d":"Periodic acid-Schiff","question":"What type of staining technique is observed in the image?"},{"a":"Eosinophilic","answer":"a","b":"Basophilic","c":"Acidophilic","d":"Neutral","question":"Which of the following best describes the predominant staining pattern in the image?"},{"a":"Nuclei","answer":"b","b":"Cytoplasm","c":"Collagen fibers","d":"Starches","question":"What cellular components are stained red in the image?"},{"a":"Necrosis","answer":"b","b":"Vascularity","c":"Presence of fat","d":"Inflammation","question":"In the context of pathology, what does the presence of red staining indicate?"},{"a":"Red","answer":"b","b":"Blue","c":"Pink","d":"Green","question":"What color does hematoxylin stain the nuclei in this image?"}],"group2":[{"a":"Adipose tissue","answer":"c","b":"Muscle tissue","c":"Vascular tissue","d":"Epithelial tissue","question":"What specific type of tissue may the red areas in the image represent?"},{"a":"Nuclei","answer":"b","b":"Fibrous tissue","c":"Stroma","d":"Lipid vesicles","question":"Which structures can be seen in the dark areas of the image?"},{"a":"Tumor presence","answer":"c","b":"Infection","c":"Inflammation","d":"Degeneration","question":"The scattered cellular components in the image imply which pathological condition?"},{"a":"Over-staining","answer":"d","b":"Under-staining","c":"Poor fixation","d":"Dehydration","question":"What staining artifact might be responsible for the black background in this image?"},{"a":"Cellular necrosis","answer":"c","b":"Foreign material","c":"Blood vessels","d":"Staining artifacts","question":"What is the significance of the yellow markers in the image?"}],"group3":[{"a":"Vascular tumor","answer":"d","b":"Chronic inflammation","c":"Granuloma","d":"Hemangioma","question":"The presence of elongated red structures in the image may suggest what specific pathology?"},{"a":"Hyperplasia","answer":"a","b":"Atrophy","c":"Dysplasia","d":"Metaplasia","question":"The distribution of staining intensity in the image can indicate what specific type of tissue response?"},{"a":"Presence of necrotic tissue","answer":"c","b":"Presence of inflammatory cells","c":"Presence of collagen deposition","d":"Presence of fibrous tissue","question":"What qualitative assessment can be made about the extracellular matrix based on the H&E staining?"},{"a":"Lymphocytes","answer":"c","b":"Eosinophils","c":"Neutrophils","d":"Basophils","question":"Which cellular morphology is indicative of an acute inflammatory response seen in the image?"},{"a":"Infectious disease","answer":"d","b":"Autoimmune disorder","c":"Malignancy","d":"Allergic reaction","question":"The features of the cells shown in the image can help diagnose which of the following conditions?"}]}
    }


def to_quiz_json(
    client, exhibit, opener,
    selected_channel_settings,
    make_quiz, model
):
    in_story = exhibit["Stories"][0]
    waypoint = in_story["Waypoints"][0]
    return to_quiz_waypoint(
        client, waypoint, opener,
        selected_channel_settings,
        make_quiz, model
    )


def update_with_llm(
    exhibit, opener, selected_channel_settings,
    make_quiz, model
):
    client = openai.AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-05-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    return to_quiz_json(
        client, exhibit, opener,
        selected_channel_settings,
        make_quiz, model
    )


def main(
    in_text, opener, roi_coordinates,
    selected_channel_settings, make_quiz
):
    model = 'gpt-4o-mini'
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
    return update_with_llm(
        in_exhibit, opener,
        selected_channel_settings,
        make_quiz, model
    )
