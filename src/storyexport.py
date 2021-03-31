import re
import os, sys
import pathlib
from distutils import file_util
from distutils.errors import DistutilsFileError

def label_to_dir(s, empty='0'):
    replaced = re.sub('[^0-9a-zA-Z _-]+', '', s).strip()
    replaced = replaced.replace(' ','_')
    replaced = replaced.replace('_','-')
    replaced = re.sub('-+', '-', replaced)
    return empty if replaced == '' else replaced

def deduplicate(data_name, data_dict, data_dir):
    """
    Return a local path for given data path
    Args:
        data_name: the basename of the target file
        data_dict: the existing mapping of local paths
        data_dir: the full path of the destination directory
    """
    n_dups = 0
    basename = data_name
    local_path = os.path.join(data_dir, basename)
    while local_path in data_dict.values():
        root, ext = os.path.splitext(basename) 
        local_path = os.path.join(data_dir, f'{root}-{n_dups}{ext}')
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
                data_name = os.path.basename(data_path)
                data_dict[data_path] = deduplicate(data_name, data_dict, data_dir)

        if 'VisBarChart' in waypoint:
            data_path = waypoint['VisBarChart']
            data_name = os.path.basename(data_path)
            data_dict[data_path] = deduplicate(data_name, data_dict, data_dir)

    return data_dict

def deduplicate_dicts(dicts, data_dir='', in_key='label', out_key='label', is_dir=False):
    """
    Map dictionaries by key to unique labels 
    Args:
        dicts: list of dicts containing input key and output key 
        data_dir: the full path of the destination directory
        in_key: used for key of output dictionary
        out_key: used for values of output dictionary
        is_dir: set true if unique labels must be directories
    """
    data_dict = dict()
    for d in dicts:
        data_in = d[in_key]
        data_name = label_to_dir(d[out_key]) if is_dir else d[out_key]
        data_dict[data_in] = deduplicate(data_name, data_dict, data_dir)

    return data_dict

def dedup_index_to_label(dicts):
    dicts_with_index = [
        {'index':i, 'label': d['label']} for (i,d) in enumerate(dicts)
    ]
    return deduplicate_dicts(dicts_with_index, '',
                            'index', 'label', False)

def dedup_index_to_path(dicts, data_dir=''):
    dicts_with_index = [
        {'index':i, 'label': d['label']} for (i,d) in enumerate(dicts)
    ]
    return deduplicate_dicts(dicts_with_index, data_dir,
                            'index', 'label', True)

def dedup_label_to_path(dicts, data_dir=''):
    return deduplicate_dicts(dicts, data_dir, 'label', 'label', True)

def mask_path_from_index(mask_data, index, data_dir=''):
    return dedup_index_to_path(mask_data, data_dir)[index]

def mask_label_from_index(mask_data, index):
    return dedup_index_to_label(mask_data)[index]

def group_path_from_label(group_data, label, data_dir=''):
    return dedup_label_to_path(group_data, data_dir)[label]

def create_story_base(title, waypoints, masks):
    """
    Creates a new minerva-story instance under subfolder named title. The subfolder will be created.
    Args:
        title: Story title, the subfolder will be named
        waypoints: List of waypoints with visData and Masks
        masks: List of masks with names and paths
    """
    out_dir = get_story_folders(title, True)[0]

    current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    export_dir = os.path.join(current_dir, title)

    try:
        # If running pyinstaller executable, _MEIPASS will contain path to the data directory in tmp
        story_dir = os.path.join(sys._MEIPASS, 'minerva-story')
    except Exception:
        # Not running pyinstaller executable; minerva-story should exist in parent directory
        story_dir = os.path.join(current_dir, '..', 'minerva-story')

    data_dir = os.path.join(export_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    try:
        file_util.copy_file(os.path.join(story_dir, 'index.html'), export_dir)
    except DistutilsFileError as e:
        print(f'Cannot copy index.html from {story_dir}')
        print(e)

    vis_path_dict = deduplicate_data(waypoints, data_dir)
    mask_index_dict = dedup_index_to_path(masks, out_dir)

    for i in range(len(masks)):
        path_i = mask_path_from_index(masks, i, out_dir)
        os.makedirs(path_i, exist_ok=True)

    for in_path, out_path in vis_path_dict.items():
        if pathlib.Path(in_path).suffix in ['.csv']:
            try:
                file_util.copy_file(in_path, out_path)
            except DistutilsFileError as e:
                print(f'Cannot copy {in_path}')
                print(e)
        else:
            print(f'Refusing to copy non-csv infovis: {in_path}')

def get_story_folders(title, create=False):
    """
    Gets paths to folders where image tiles, json, dat-file and log file must be saved.
    Args:
        title: Story title
        create: Whether folders should be created

    Returns: Tuple of images dir, json config dir, json save dir, log dir
    """
    folder = os.path.dirname(os.path.abspath(sys.argv[0]))
    images_folder = os.path.join(folder, title, 'images')
    out_dir = os.path.join(images_folder, title)

    out_json_config = os.path.join(folder, title, 'exhibit.json')

    out_json_save = os.path.join(folder, title + '.json')

    # After version 1.6.0 use .story.json, keep support for existing files
    if not os.path.exists(out_json_save):
        out_json_save = os.path.join(folder, title + '.story.json')

    out_log = os.path.join(folder, title + '.log')

    if create:
        os.makedirs(images_folder, exist_ok=True)

    return out_dir, out_json_config, out_json_save, out_log
