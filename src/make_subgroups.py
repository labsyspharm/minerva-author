def compare(chan, render):
    cmin = render["min"]
    cmax = render["max"]
    return f"{chan}-{cmin:1.4f}-{cmax:1.4f}"

def make_subgroups(d, rgba):
    used = set()
    for group in d:
        channels = group["channels"]
        renders = group.get("render", channels)
        if rgba:
            yield group
        for channel, render in zip(channels, renders):
            ckey = compare(channel["id"], render)
            if ckey in used: continue
            used.add(ckey)
            yield {
                "render": [render],
                "channels": [channel],
                "label": channel["label"],
            }
