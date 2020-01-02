import struct

def is_bigtiff(filename):
    """Check if a tiff image is bigtiff or not.

    Args:
        filename (str): Filename of a tiff image.

    Returns:
        bool: True if the file is a bigtiff, False otherwise.
    """
    with open(filename, "rb") as handle:
        byteorder = {b'II': '<', b'MM': '>'}[handle.read(2)]
        version = struct.unpack(byteorder + 'H', handle.read(2))[0]
        if version == 43:
            return True
        else:
            return False

def byteorder(filename):
    """Check the byteorder of a given file.

    Args:
        filename (str): Filename of a tiff image.

    Returns:
        str: '<' for little endian, '>' for big endian.
    """
    with open(filename, "rb") as handle:
        tmp = {b'II': '<', b'MM': '>'}[handle.read(2)]
        return tmp
