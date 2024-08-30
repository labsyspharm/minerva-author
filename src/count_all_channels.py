from make_subgroups import make_subgroups
import json
import sys

def from_json(fname, rgba):
    with open(fname, "r") as fh:
        data = json.load(fh)
        single_channel_groups = list(make_subgroups(
            data['groups'], rgba
        ))
        return len(single_channel_groups)

if __name__ == "__main__":
    fname = sys.argv[1]
    rgba = len(sys.argv) > 2 and sys.argv[2] == 'rgba'
    count  = from_json(fname, rgba)
    print(int(count))
