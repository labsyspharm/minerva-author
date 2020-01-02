from pytiff import byteorder, is_bigtiff

def test_byteorder():
    assert byteorder("test_data/small_example.tif") == "<"
    assert byteorder("test_data/big_endian_small_example.tif") == ">"

def test_is_bigtiff():
    assert not is_bigtiff("test_data/small_example.tif")
    assert is_bigtiff("test_data/bigtif_example.tif")
