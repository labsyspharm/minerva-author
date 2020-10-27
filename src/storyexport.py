import os, sys
from distutils import file_util

def deduplicate_path(data_path, data_dict, data_dir):
    """
    Return a local path for given data path
    Args:
        data_path: the full path of the file to copy
        data_dict: the existing mapping of local paths
        data_dir: the full path of the destination directory
    """
    n_dups = 0
    basename = os.path.basename(data_path)
    local_path = os.path.join(data_dir, basename)
    while local_path in data_dict.values():
        root, ext = os.path.splitext(basename) 
        basename = f'{root}_{n_dups}{ext}'
        local_path = os.path.join(data_dir, basename)
        n_dups += 1
    return local_path

def deduplicate_data(waypoints, data_dir):
    """
    Map filesystem paths to local data paths
    Args:
        waypoints: list of dicts containing optional VisData keys
        data_dir: the full path of the destination directory
    """
    data_dict = dict()
    for waypoint in waypoints:
        for vis in ['VisScatterplot', 'VisCanvasScatterplot', 'VisMatrix']:
            if vis in waypoint:
                data_path = waypoint[vis]['data']
                data_dict[data_path] = deduplicate_path(data_path, data_dict, data_dir)

        if 'VisBarChart' in waypoint:
            data_dict[data_path] = deduplicate_path(waypoint['VisBarChart'], data_dict, data_dir)

    return data_dict

def create_story_base(title, waypoints):
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

    data_dir = os.path.join(export_dir, 'data')
    images_dir = os.path.join(export_dir, 'images')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    file_util.copy_file(os.path.join(story_dir, 'index.html'), export_dir)
    file_util.copy_file(os.path.join(story_dir, 'bundle.js'), export_dir)

    data_dict = deduplicate_data(waypoints, data_dir)

    for waypoint in waypoints:
        for vis in ['VisScatterplot', 'VisCanvasScatterplot', 'VisMatrix']:
            if vis in waypoint:
                data_path = waypoint[vis]['data']
                file_util.copy_file(data_path, data_dict[data_path])

        if 'VisBarChart' in waypoint:
            file_util.copy_file(data_path, data_dict[waypoint['VisBarChart']])

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
    images_folder = os.path.join(folder, out_name, 'images')
    out_dir = os.path.join(images_folder, out_name)

    out_json_config = os.path.join(folder, out_name, 'exhibit.json')

    out_json_save = os.path.join(folder, out_name + '.json')
    out_log = os.path.join(folder, out_name + '.log')

    if create:
        os.makedirs(images_folder, exist_ok=True)

    return out_dir, out_json_config, out_json_save, out_log
