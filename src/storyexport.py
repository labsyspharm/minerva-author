import os, sys
from distutils import dir_util


def create_story_base(title):
    """
    Creates a new minerva-story instance under subfolder named title. The subfolder will be created.
    Args:
        title: Story title, the subfolder will be named
    """
    get_story_folders(title, True)

    current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    export_dir = os.path.join(current_dir, title)

    try:
        # If running pyinstaller executable, _MEIPASS will contain path to the data directory in tmp
        story_dir = os.path.join(sys._MEIPASS, 'minerva-story')
    except Exception:
        # Not running pyinstaller executable; minerva-story should exist in parent directory
        story_dir = os.path.join(current_dir, '..', 'minerva-story')

    exported_story_dir = os.path.join(export_dir, 'minerva-story')
    os.makedirs(exported_story_dir, exist_ok=True)

    dir_util.copy_tree(story_dir, exported_story_dir)

def get_story_folders(title, create=False):
    """
    Gets paths to folders where image tiles, json, dat-file and log file must be saved.
    Args:
        title: Story title
        create: Whether folders should be created

    Returns: Tuple of images dir, json config dir, json save dir, log dir
    """
    out_name = title.replace(' ', '_')
    folder = os.path.dirname(os.path.abspath(sys.argv[0]))
    images_folder = os.path.join(folder, out_name, 'minerva-story', 'images')

    out_dir = os.path.join(images_folder, out_name)

    includes_folder = os.path.join(folder, out_name, 'minerva-story', 'includes')
    out_json_config = os.path.join(includes_folder, 'exhibit.json')

    out_json_save = os.path.join(folder, out_name + '.json')
    out_log = os.path.join(folder, out_name + '.log')

    if create:
        os.makedirs(images_folder, exist_ok=True)

    return out_dir, out_json_config, out_json_save, out_log
