import os, sys
from distutils import dir_util

def create_story_base(title):
    get_story_folders(title, True)

    current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    export_dir = os.path.join(current_dir, title)

    try:
        # If using pyinstaller executable, _MEIPASS will contain path to the data directory in tmp
        story_dir = os.path.join(sys._MEIPASS, 'minerva-story')
    except Exception:
        story_dir = os.path.join(current_dir, '..', 'minerva-story')

    exported_story_dir = os.path.join(export_dir, 'minerva-story')
    os.makedirs(exported_story_dir, exist_ok=True)

    dir_util.copy_tree(story_dir, exported_story_dir)

    markdown_path = os.path.join(current_dir, title, 'minerva-story', 'index.md')
    markdown = f"""---
layout: osd-exhibit
paper: {title}
figure: {title}
---"""
    with open(markdown_path, "w") as f:
        f.write(markdown)

def get_story_folders(title, create=False):
    out_name = title.replace(' ', '_')
    folder = os.path.dirname(os.path.abspath(sys.argv[0]))
    images_folder = os.path.join(folder, out_name, 'minerva-story', 'images')

    out_dir = os.path.join(images_folder, out_name)

    yaml_folder = os.path.join(folder, out_name, 'minerva-story', '_data', out_name)
    out_yaml = os.path.join(yaml_folder, out_name + '.yaml')

    out_dat = os.path.join(folder, out_name + '.dat')

    if create:
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(yaml_folder, exist_ok=True)

    return out_dir, out_yaml, out_dat
