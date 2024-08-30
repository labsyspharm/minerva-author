#from app import Opener 
from make_subgroups import make_subgroups
import json
import sys

def from_json(fname, idx, rgba):
    with open(fname, "r") as fh:
        data = json.load(fh)
        single_channel_groups = list(make_subgroups(
            data['groups'], rgba
        ))
        group = single_channel_groups[idx]
        data['defaults'] = group['render']
        data['groups'] = [group]
        data['waypoints'] = []
        if 'autosave' in data:
            del data['autosave']
        return data

if __name__ == "__main__":
    fname = sys.argv[1]
    idx = int(sys.argv[2])
    rgba = len(sys.argv) > 3 and sys.argv[3] == 'rgba'
    idx_channel_group = from_json(fname, idx, rgba)
    print(json.dumps(idx_channel_group))
