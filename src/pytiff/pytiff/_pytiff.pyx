#cython: c_string_type=str, c_string_encoding=ascii
"""
pytiff is a python wrapper for the libtiff c api written in cython. It is python 2 and 3 compatible.
While there are some missing features, it supports reading chunks of tiled greyscale tif images as well as basic reading for color images.
Apart from that multipage tiffs are supported. It also supports writing greyscale images in tiles or scanlines.
"""

cimport ctiff
from libcpp.string cimport string
import logging
from cpython cimport bool
cimport numpy as np
import numpy as np
from math import ceil
import re
from pytiff._version import _package
import sys
import copy
from enum import IntEnum
PY3 = sys.version_info[0] == 3

TYPE_MAP = {
  1: {
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64
  },
  2: {
    8: np.int8,
    16: np.int16,
    32: np.int32,
    64: np.int64
  },
  3: {
    8: None,
    16: np.float16,
    32: np.float32,
    64: np.float64
  }
}

# map data type to (sample_format, bitspersample)
INVERSE_TYPE_MAP = {
  np.dtype('uint8'): (1, 8),
  np.dtype('uint16'): (1, 16),
  np.dtype('uint32'): (1, 32),
  np.dtype('uint64'): (1, 64),
  np.dtype('int8'): (2, 8),
  np.dtype('int16'): (2, 16),
  np.dtype('int32'): (2, 32),
  np.dtype('int64'): (2, 64),
  np.dtype('float16'): (3, 16),
  np.dtype('float32'): (3, 32),
  np.dtype('float64'): (3, 64)
}

# map tiff_tags to attribute name and type
# code: (attribute name, default value, type, count, validator)
TIFF_TAGS = {
    254: ('new_subfile_type', 0, 4, 1),
    #255: ('subfile_type', None, 3, 1), should not be used
    256: ('image_width', None, 4, 1),
    257: ('image_length', None, 4, 1),
    258: ('bits_per_sample', 1, 3, 1), # changed from None to 1 since libtiff only support same number of bits per sample
    259: ('compression', 1, 3, 1),
    262: ('photometric', None, 3, 1),
    266: ('fill_order', 1, 3, 1),
    269: ('document_name', None, 2, None),
    270: ('image_description', None, 2, None),
    271: ('make', None, 2, None),
    272: ('model', None, 2, None),
    273: ('strip_offsets', None, 16, None),
    274: ('orientation', 1, 3, 1),
    277: ('samples_per_pixel', 1, 3, 1),
    278: ('rows_per_strip', 2**32-1, 4, 1),
    279: ('strip_byte_counts', None, 16, None),
    280: ('min_sample_value', None, 3, None),
    281: ('max_sample_value', None, 3, None),  # 2**bits_per_sample
    282: ('x_resolution', None, 11, 1), # change from type 5 (rational) to float
    283: ('y_resolution', None, 11, 1),  # change from type 5 (rational) to float
    284: ('planar_configuration', 1, 3, 1),
    285: ('page_name', None, 2, None),
    286: ('x_position', None, 5, 1),
    287: ('y_position', None, 5, 1),
    296: ('resolution_unit', 2, 4, 1),
    297: ('page_number', None, 3, 2),
    305: ('software', None, 2, None),
    306: ('datetime', None, 2, None),
    315: ('artist', None, 2, None),
    316: ('host_computer', None, 2, None),
    317: ('predictor', 1, 3, 1),
    318: ('white_point', None, 5, 2),
    319: ('primary_chromaticities', None, 5, 6),
    #320: ('color_map', None, 3, None),
    322: ('tile_width', None, 4, 1),
    323: ('tile_length', None, 4, 1),
    324: ('tile_offsets', None, 16, None),
    325: ('tile_byte_counts', None, 16, None),
    #330: ('sub_ifds', None, 4, None),
    338: ('extra_samples', None, 3, None),
    339: ('sample_format', 1, 3, 1), # changed None to 1
    340: ('smin_sample_value', None, None, None),
    341: ('smax_sample_value', None, None, None),
    346: ('indexed', 0, 3, 1),
    #347: ('jpeg_tables', None, 7, None),
    530: ('ycbcr_subsampling', (1, 1), 3, 2),
    531: ('ycbcr_positioning', (1, 1), 3, 1),
    532: ('reference_black_white', None, 5, 1),
    32995: ('sgi_matteing', None, None, 1),  # use extra_samples
    #32996: ('sgi_datatype', None, None, None),  # use sample_format
    32997: ('image_depth', 1, 4, 1),
    32998: ('tile_depth', None, 4, 1),
    #33432: ('copyright', None, 1, None),
    33445: ('md_file_tag', None, 4, 1),
    33446: ('md_scale_pixel', None, 5, 1),
    #33447: ('md_color_table', None, 3, None),
    33448: ('md_lab_name', None, 2, None),
    33449: ('md_sample_info', None, 2, None),
    33450: ('md_prep_date', None, 2, None),
    33451: ('md_prep_time', None, 2, None),
    33452: ('md_file_units', None, 2, None),
    33550: ('model_pixel_scale', None, 12, 3),
    #33922: ('model_tie_point', None, 12, None),
    34665: ('exif_ifd', None, None, 1),
    #34735: ('geo_key_directory', None, 3, None),
    #34736: ('geo_double_params', None, 12, None),
    34737: ('geo_ascii_params', None, 2, None),
    34853: ('gps_ifd', None, None, 1),
    #37510: ('user_comment', None, None, None),
    #42112: ('gdal_metadata', None, 2, None),
    #42113: ('gdal_nodata', None, 2, None),
    50289: ('mc_xy_position', None, 12, 2),
    50290: ('mc_z_position', None, 12, 1),
    50291: ('mc_xy_calibration', None, 12, 3),
    50292: ('mc_lens_lem_na_n', None, 12, 3),
    #50293: ('mc_channel_name', None, 1, None),
    50294: ('mc_ex_wavelength', None, 12, 1),
    50295: ('mc_time_stamp', None, 12, 1),
    #50838: ('imagej_byte_counts', None, None, None),
    51023: ('fibics_xml', None, 2, None),
    65200: ('flex_xml', None, 2, None),
    # code: (attribute name, default value, type, count, validator)
}

# attribute_name: tag
TIFF_TAGS_REVERSE = {
    'new_subfile_type':           254,
    'subfile_type':               255,
    'image_width':                256,
    'image_length':               257,
    'bits_per_sample':            258,
    'compression':                259,
    'photometric':                262,
    'fill_order':                 266,
    'document_name':              269,
    'image_description':          270,
    'make':                       271,
    'model':                      272,
    'strip_offsets':              273,
    'orientation':                274,
    'samples_per_pixel':          277,
    'rows_per_strip':             278,
    'strip_byte_counts':          279,
    'min_sample_value':           280,
    'max_sample_value':           281,
    'x_resolution':               282,
    'y_resolution':               283,
    'planar_configuration':       284,
    'page_name':                  285,
    'x_position':                 286,
    'y_position':                 287,
    'resolution_unit':            296,
    'page_number':                297,
    'software':                   305,
    'datetime':                   306,
    'artist':                     315,
    'host_computer':              316,
    'predictor':                  317,
    'white_point':                318,
    'primary_chromaticities':     319,
    'color_map':                  320,
    'tile_width':                 322,
    'tile_length':                323,
    'tile_offsets':               324,
    'tile_byte_counts':           325,
    'sub_ifds':                   330,
    'extra_samples':              338,
    'sample_format':              339,
    'smin_sample_value':          340,
    'smax_sample_value':          341,
    'indexed':                    346,
    'jpeg_tables':                347,
    'ycbcr_subsampling':          530,
    'ycbcr_positioning':          531,
    'reference_black_white':      532,
    'sgi_matteing':             32995,
    'sgi_datatype':             32996,
    'image_depth':              32997,
    'tile_depth':               32998,
    'copyright':                33432,
    'md_file_tag':              33445,
    'md_scale_pixel':           33446,
    'md_color_table':           33447,
    'md_lab_name':              33448,
    'md_sample_info':           33449,
    'md_prep_date':             33450,
    'md_prep_time':             33451,
    'md_file_units':            33452,
    'model_pixel_scale':        33550,
    'model_tie_point':          33922,
    'exif_ifd':                 34665,
    'geo_key_directory':        34735,
    'geo_double_params':        34736,
    'geo_ascii_params':         34737,
    'gps_ifd':                  34853,
    'user_comment':             37510,
    'gdal_metadata':            42112,
    'gdal_nodata':              42113,
    'mc_xy_position':           50289,
    'mc_z_position':            50290,
    'mc_xy_calibration':        50291,
    'mc_lens_lem_na_n':         50292,
    'mc_channel_name':          50293,
    'mc_ex_wavelength':         50294,
    'mc_time_stamp':            50295,
    'imagej_byte_counts':       50838,
    'fibics_xml':               51023,
    'flex_xml':                 65200,
    # code: (attribute name, default value, type, count, validator)
}

tags = IntEnum("tags", names=TIFF_TAGS_REVERSE)

TIFF_TAGS_NOT_WRITABLE = [
        tags["tile_offsets"],
        tags["tile_byte_counts"],
        tags["strip_byte_counts"],
        tags["strip_offsets"]
        ]

# the data types to the corresponding type in TIFF_TAGS
TIFF_DATA_TYPES = {
    1: np.dtype("uint8"),       # BYTE 8-bit unsigned integer.
    2: np.dtype("uint64"),      # ASCII 8-bit byte that contains a 7-bit ASCII code;
                                #   the last byte must be NULL (binary zero).
    3: np.dtype("uint16"),      # SHORT 16-bit (2-byte) unsigned integer
    4: np.dtype("uint32"),      # LONG 32-bit (4-byte) unsigned integer.
    5: np.dtype("uint64"),      # RATIONAL Two LONGs: the first represents the numerator of
                                #   a fraction; the second, the denominator.
    6: np.dtype("int8"),        # SBYTE An 8-bit signed (twos-complement) integer.
    7: np.dtype("uint8"),      # UNDEFINED An 8-bit byte that may contain anything,
                                #   depending on the definition of the field.
    8: np.dtype("int16"),       # SSHORT A 16-bit (2-byte) signed (twos-complement) integer.
    9: np.dtype("int32"),       # SLONG A 32-bit (4-byte) signed (twos-complement) integer.
    10: np.dtype("int64"),      # SRATIONAL Two SLONGs: the first represents the numerator
                                #   of a fraction, the second the denominator.
    11: np.dtype("float32"),    # FLOAT Single precision (4-byte) IEEE format.
    12: np.dtype("float64"),    # DOUBLE Double precision (8-byte) IEEE format.
    13: np.dtype("uint32"),     # IFD unsigned 4 byte IFD offset.
    # 14: '',                   # UNICODE
    # 15: '',                   # COMPLEX
    16: np.dtype("uint64"),     # LONG8 unsigned 8 byte integer (BigTiff)
    17: np.dtype("int64"),      # SLONG8 signed 8 byte integer (BigTiff)
    18: np.dtype("uint64"),     # IFD8 unsigned 8 byte IFD offset (BigTiff)
}

cdef _to_view(void* pointer, dtype, size):
    cdef np.ndarray ar
    if dtype == np.dtype("uint8"):
        ar = np.zeros(size, dtype)
        for i in range(size):
            ar[i] = (<unsigned char*> pointer)[i]
    elif dtype == np.dtype("uint16"):
        ar = np.zeros(size, dtype)
        for i in range(size):
            ar[i] = (<unsigned short*> pointer)[i]
    elif dtype == np.dtype("uint32"):
        ar = np.zeros(size, dtype)
        for i in range(size):
            ar[i] = (<unsigned int*> pointer)[i]
    elif dtype == np.dtype("uint64"):
        # memory view raises an error TypeError('expected bytes, str found')
        ar = np.zeros(size, dtype)
        for i in range(size):
            ar[i] = (<unsigned long*> pointer)[i]
    elif dtype == np.dtype("int8"):
        ar = np.zeros(size, dtype)
        for i in range(size):
            ar[i] = (<char*> pointer)[i]
    elif dtype == np.dtype("int16"):
        ar = np.zeros(size, dtype)
        for i in range(size):
            ar[i] = (<short*> pointer)[i]
    elif dtype == np.dtype("int32"):
        ar = np.zeros(size, dtype)
        for i in range(size):
            ar[i] = (<int*> pointer)[i]
    elif dtype == np.dtype("int64"):
        ar = np.zeros(size, dtype)
        for i in range(size):
            ar[i] = (<long*> pointer)[i]
    elif dtype == np.dtype("float32"):
        ar = np.zeros(size, dtype)
        for i in range(size):
            ar[i] = (<float*> pointer)[i]
    elif dtype == np.dtype("float64"):
        ar = np.zeros(size, dtype)
        for i in range(size):
            ar[i] = (<double*> pointer)[i]
    return ar

cdef unsigned int MIN_IS_BLACK = 1
cdef unsigned int MIN_IS_WHITE = 0
cdef unsigned int NO_COMPRESSION = 1
cdef unsigned int RGB = 2

def tiff_version_raw():
  """Return the raw version string of libtiff."""
  return ctiff.TIFFGetVersion()

def tiff_version():
  """Parse the version of libtiff and return it."""
  cdef string str_version = tiff_version_raw()
  m = re.search("(?<=[Vv]ersion )\d+\.\d+\.?\d*", str_version)
  return m.group(0)

class NotTiledError(Exception):
  def __init__(self, message):
    self.message = message

class SinglePageError(Exception):
  def __init__(self):
      self.message = "Changing pages is disabled for this object."

cdef _get_rgb(np.ndarray[np.uint32_t, ndim=2] inp, short n_samples):
  shape = (inp.shape[0], inp.shape[1], n_samples)
  cdef np.ndarray[np.uint8_t, ndim=3] rgb = np.zeros(shape, np.uint8)

  cdef unsigned long int row, col
  for row in range(shape[0]):
    for col in range(shape[1]):
      rgb[row, col, 0] = ctiff.TIFFGetR(inp[row, col])
      rgb[row, col, 1] = ctiff.TIFFGetG(inp[row, col])
      rgb[row, col, 2] = ctiff.TIFFGetB(inp[row, col])
      # add alpha channel if more than 3 samples
      if n_samples > 3:
        rgb[row, col, 3] = ctiff.TIFFGetA(inp[row, col])

  return rgb

cpdef object rebuild(data):
    filename, file_mode, bigtiff, current_page = data
    obj = Tiff(filename, file_mode, bigtiff)
    obj.set_page(current_page)
    return obj


cdef class Tiff:
  """The Tiff class handles tiff files.

  The class is able to read chunked greyscale images as well as basic reading of color images.
  Currently writing tiff files is not supported.

  Examples:
    >>> with pytiff.Tiff("tiff_file.tif") as f:
    >>>   chunk = f[100:300, 50:100]
    >>>   print(type(chunk))
    >>>   print(chunk.shape)
    numpy.ndarray
    (200, 50)

  Args:
    filename (string): The filename of the tiff file.
    file_mode (string): File mode either "w" for writing (old data is deleted), "a" for appending or "r" for reading. Default: "r".
    bigiff (bool): If True the file is assumed to be bigtiff. Default: False.
  """
  cdef ctiff.TIFF* tiff_handle
  cdef public short samples_per_pixel
  cdef short[:] n_bits_view
  cdef unsigned short[:] extra_samples
  cdef short sample_format, n_pages, _write_mode_n_pages
  cdef bool closed, cached, _unsaved_page
  cdef unsigned int image_width, image_length, tile_width, tile_length
  cdef object cache, logger
  cdef public object filename
  cdef object file_mode
  cdef public object tags
  cdef _dtype_write
  cdef object _singlepage
  cdef object _pages

  def __cinit__(self, filename, file_mode="r", bigtiff=False):
    if bigtiff:
      file_mode += "8"
    tmp_filename = <string> filename
    tmp_mode = <string> file_mode
    self.closed = True
    self.filename = tmp_filename
    self.file_mode = tmp_mode
    self._write_mode_n_pages = 0
    self.n_pages = 0
    self._singlepage = False
    self._pages = None
    self.tiff_handle = ctiff.TIFFOpen(tmp_filename.c_str(), tmp_mode.c_str())
    if self.tiff_handle is NULL:
      raise IOError("file not found!")
    self.closed = False
    self._unsaved_page = False

    self.logger = logging.getLogger(_package)
    self.logger.debug("Tiff object created. file: {}".format(filename))
    cdef np.ndarray[np.int16_t, ndim=1] write_pages_buffer = np.zeros(2, dtype=np.int16)
    if self.file_mode == "r":
      self._init_page()

  def _init_page(self):
    """Initialize page specific attributes."""
    self.logger.debug("_init_page called.")
    self.samples_per_pixel = 1
    err = ctiff.TIFFGetField(self.tiff_handle, tags.samples_per_pixel, &self.samples_per_pixel)
    if err != 1:
        self.logger.warn("[FAIL] Could not read samples per pixel tag! 1 is assumed!")
        self.samples_per_pixel = 1
    self.logger.debug("[SUCCESS] read samples per pixel: {}".format(self.samples_per_pixel))
    cdef np.ndarray[np.int16_t, ndim=1] bits_buffer = np.zeros(self.samples_per_pixel, dtype=np.int16)
    err = ctiff.TIFFGetField(self.tiff_handle, tags.bits_per_sample, <ctiff.ttag_t*>bits_buffer.data)
    if err != 1:
        self.logger.warn("[FAIL] Could not read bits per sample tag!")
    self.n_bits_view = bits_buffer
    self.logger.debug("[SUCCESS] read bits per sample")

    self.sample_format = 1
    ctiff.TIFFGetField(self.tiff_handle, tags.sample_format, &self.sample_format)
    self.logger.debug("[SUCCESS] read sample format")

    ctiff.TIFFGetField(self.tiff_handle, tags.image_width, &self.image_width)
    self.logger.debug("[SUCCESS] read image width")
    ctiff.TIFFGetField(self.tiff_handle, tags.image_length, &self.image_length)
    self.logger.debug("[SUCCESS] read image length")

    ctiff.TIFFGetField(self.tiff_handle, tags.tile_width, &self.tile_width)
    self.logger.debug("[SUCCESS] read tile width")
    ctiff.TIFFGetField(self.tiff_handle, tags.tile_length, &self.tile_length)
    self.logger.debug("[SUCCESS] read tile length")

    # get extra samples
    cdef unsigned short* _extra = NULL
    cdef unsigned short nextra;

    err = ctiff.TIFFGetField(self.tiff_handle, tags.extra_samples, &nextra, &_extra)
    self.logger.debug("[SUCCESS] read extra samples #{}, err: {}".format(nextra, err))
    if err == 1:
        self.extra_samples = np.zeros(nextra, dtype=np.uint16)
        for i in range(nextra):
            self.extra_samples[i] = _extra[i]
    else:
        self.extra_samples = np.zeros(0, dtype=np.uint16)
    self.logger.debug("[SUCCESS] read extra samples {}".format(np.asarray(self.extra_samples)))

    # read tags for new page
    self.read_tags()
    self.cached = False

  def __reduce__(self):
      if "w" in self.file_mode or "a" in self.file_mode:
          self.logger.warn("Tiff Object is pickled in write or append mode")

      bigtiff = False
      if "8" in self.file_mode:
          bigtiff = True
      data = self.filename, self.file_mode, bigtiff, self.current_page
      return rebuild, (data,)

  @property
  def description(self):
    """Returns the image description. If not available, returns None."""
    if tags.image_description in self.tags:
        desc = self.tags[tags.image_description]
    else:
        desc = None
    return desc

  def close(self):
    """Close the filehandle."""
    if not self.closed:
      self.logger.debug("Closing file manually. file: {}".format(self.filename))
      if self._pages is not None:
        for p in self._pages:
            p.close()
      ctiff.TIFFClose(self.tiff_handle)
      self.closed = True
      return

  def __dealloc__(self):
    if not self.closed:
      self.logger.debug("Closing file automatically. file: {}".format(self.filename))
      if self._pages is not None:
          for p in self._pages:
              p.close()
      ctiff.TIFFClose(self.tiff_handle)

  @property
  def mode(self):
    """Mode of the current image. Can either be 'rgb' or 'greyscale'.

    'rgb' is returned if the sampels per pixel are larger than 1. This means 'rgb' is always returned
    if the image is not 'greyscale'.
    """
    if self.samples_per_pixel > 1:
      return "rgb"
    else:
      return "greyscale"

  @property
  def size(self):
    """Returns a tuple with the current image size.

    size is equal to numpys shape attribute.

    Returns:
      tuple: `(image height, image width)`

      This is equal to:
      `(number_of_rows, number_of_columns)`
    """
    size = self.image_length, self.image_width

    if self.samples_per_pixel > 1:
        size += (self.samples_per_pixel,)
    return size

  @property
  def shape(self):
    """The shape property is an alias for the size property."""
    return self.size

  @property
  def n_bits(self):
    """Returns an array with the bit size for each sample of a pixel."""
    return np.array(self.n_bits_view)

  @property
  def dtype(self):
    """Maps the image data type to an according numpy type.

    Returns:
      type: numpy dtype of the image.

      If the mode is 'rgb', the dtype is always uint8. Most times a rgb image is saved as a
      uint32 array. One value is containing all four values of an RGBA image. Thus the dtype of the numpy array
      is uint8.

      If the mode is 'greyscale', the dtype is the type of the first sample.
      Since greyscale images only have one sample per pixel, this resembles the general dtype.
    """
    if "a" in self.file_mode or "w" in self.file_mode:
      return self._dtype_write.type
    if self.mode == "rgb":
      self.logger.debug("RGB Image assumed for dtype.")
      return np.uint8
    return TYPE_MAP[self.sample_format][self.n_bits[0]]

  @property
  def current_page(self):
    """Current page/directory of the tiff file.

    Returns:
      int: index of the current page/directory.
    """
    return ctiff.TIFFCurrentDirectory(self.tiff_handle)

  def set_page(self, value):
    """Set the page/directory of the tiff file.

    Args:
      value (int): page index
    """
    if self._singlepage:
        raise SinglePageError()
    ctiff.TIFFSetDirectory(self.tiff_handle, value)
    self._init_page()

  @property
  def number_of_pages(self):
    """number of pages/directories in the tiff file.

    Returns:
      int: number of pages/directories
    """
    # dont use
    # fails if only one directory
    # ctiff.TIFFNumberOfDirectories(self.tiff_handle)
    if self.file_mode == "r":
      return self._number_of_pages_readmode()
    else:
      return self._number_of_pages_writemode()

  def _number_of_pages_readmode(self):
    current_dir = self.current_page
    if self.n_pages != 0:
      return self.n_pages
    else:
      cont = 1
      while cont:
        self.n_pages += 1
        cont = ctiff.TIFFReadDirectory(self.tiff_handle)
      ctiff.TIFFSetDirectory(self.tiff_handle, current_dir)
    return self.n_pages

  def _number_of_pages_writemode(self):
    return self._write_mode_n_pages

  @property
  def n_samples(self):
    cdef short samples_in_file = self.samples_per_pixel - self.extra_samples.size
    return samples_in_file

  def is_tiled(self):
    """Return True if image is tiled, else False."""
    tiled = ctiff.TIFFIsTiled(self.tiff_handle)
    if tiled > 0:
        return True
    else:
        return False

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.close()

  def _load_all(self):
    """Load the image at once.

    If n_samples > 1 a rgba image is returned, else a greyscale image is assumed.

    Returns:
      array_like: RGBA image (3 dimensions) or Greyscale (2 dimensions)
    """
    if self.cached:
      return self.cache
    if self.n_samples > 1:
      data = self._load_all_rgba()
    else:
      data = self._load_all_grey()

    self.cache = data
    self.cached = True
    return data

  def _load_all_rgba(self):
    """Loads an image at once. Returns an RGBA image."""
    self.logger.debug("Loading a whole rgba image.")
    cdef np.ndarray buffer
    shape = self.size[:2]
    buffer = np.zeros(shape, dtype=np.uint32)
    ctiff.TIFFReadRGBAImage(self.tiff_handle, self.image_width, self.image_length, <unsigned int*>buffer.data, 0)
    rgb = _get_rgb(buffer, self.samples_per_pixel)
    rgb = np.flipud(rgb)
    return rgb

  def _load_all_grey(self):
    """Loads an image at once. Returns a greyscale image."""
    self.logger.debug("Loading a whole greyscale image.")
    cdef np.ndarray total = np.zeros(self.size, dtype=self.dtype)
    cdef np.ndarray buffer = np.zeros(self.image_width, dtype=self.dtype)

    for i in range(self.image_length):
      ctiff.TIFFReadScanline(self.tiff_handle,<void*> buffer.data, i, 0)
      total[i] = buffer
    return total

  def _load_tiled(self, y_range, x_range):
    self.logger.debug("Loading tiled image. RGBA is assumed as RGBA,RGBA... for each pixel.")
    cdef unsigned int z_size, start_x, start_y, start_x_offset, start_y_offset
    cdef unsigned int end_x, end_y, end_x_offset, end_y_offset
    if not self.tile_width:
      raise NotTiledError("Image is not tiled!")

    # use rgba if no greyscale image
    z_size = self.n_samples

    shape = (y_range[1] - y_range[0], x_range[1] - x_range[0], z_size)

    start_x = x_range[0] // self.tile_width
    start_y = y_range[0] // self.tile_length
    end_x = ceil(float(x_range[1]) / self.tile_width)
    end_y = ceil(float(y_range[1]) / self.tile_length)
    offset_x = start_x * self.tile_width
    offset_y = start_y * self.tile_length

    large = (end_y - start_y) * self.tile_length, (end_x - start_x) * self.tile_width, z_size

    self.logger.debug("loading tiled, dtype: {}".format(self.dtype))
    cdef np.ndarray large_buf = np.zeros(large, dtype=self.dtype).squeeze()
    cdef np.ndarray arr_buf = np.zeros(shape, dtype=self.dtype).squeeze()
    self.logger.debug("large_buf dtype: {}".format(large_buf.dtype))
    self.logger.debug("arr_buf dtype: {}".format(arr_buf.dtype))
    cdef unsigned int np_x, np_y
    np_x = 0
    np_y = 0
    for current_y in np.arange(start_y, end_y):
      np_x = 0
      for current_x in np.arange(start_x, end_x):
        real_x = current_x * self.tile_width
        real_y = current_y * self.tile_length
        tmp = self._read_tile(real_y, real_x)
        e_x = np_x + tmp.shape[1]
        e_y = np_y + tmp.shape[0]

        large_buf[np_y:e_y, np_x:e_x] = tmp
        np_x += self.tile_width

      np_y += self.tile_length

    arr_buf = large_buf[y_range[0]-offset_y:y_range[1]-offset_y, x_range[0]-offset_x:x_range[1]-offset_x]
    return arr_buf

  def _get(self, y_range=None, x_range=None):
    """Function to load a chunk of an image.

    Should not be used. Instead use numpy style slicing.

    Examples:
      >>> with pytiff.Tiff("tiffile.tif") as f:
      >>>   total = f[:, :] # f[:]
      >>>   part = f[100:200,:]
    """

    if x_range is None:
      x_range = (0, self.image_width)
    if y_range is None:
      y_range = (0, self.image_length)

    cdef np.ndarray res, tmp
    try:
      res = self._load_tiled(y_range, x_range)
    except NotTiledError as e:
      self.logger.debug(e.message)
      self.logger.debug("Warning: chunks not available! Loading all data!")
      tmp = self._load_all()
      res = tmp[y_range[0]:y_range[1], x_range[0]:x_range[1]]

    return res

  def __getitem__(self, index):
    self.logger.debug("__getitem__ called")
    if not isinstance(index, tuple):
      if isinstance(index, slice):
        index = (index, slice(None,None,None))
      else:
        raise Exception("Only slicing is supported")
    elif len(index) < 3:
      index = index[0],index[1],0

    if not isinstance(index[0], slice) or not isinstance(index[1], slice):
      raise Exception("Only slicing is supported")

    x_range = np.array((index[1].start, index[1].stop))
    if x_range[0] is None:
      x_range[0] = 0
    if x_range[1] is None or x_range[1] > self.image_width:
      x_range[1] = self.image_width

    y_range = np.array((index[0].start, index[0].stop))
    if y_range[0] is None:
      y_range[0] = 0
    if y_range[1] is None or y_range[1] > self.image_length:
      y_range[1] = self.image_length

    return self._get(y_range, x_range)

  def __array__(self, dtype=None):
    return self.__getitem__(slice(None))

  def write(self, np.ndarray data, **options):
    """Write data to the tif file.

    If the file is opened in write mode, a numpy array can be written to a
    tiff page.
    Multipage tiffs are supperted by calling write multiple times.

    Args:
        data (array_like): 2D numpy array. Supported dtypes: un(signed) integer, float.
        method: determines which method is used for writing. Either "tile" for tiled tiffs or "scanline" for basic scanline tiffs. Default: "tile"
        photometric: determines how values are interpreted, either zero == black or zero == white.
                     MIN_IS_BLACK(default), MIN_IS_WHITE. more information can be found in the libtiff doc.
        planar_config: defaults to 1, component values for each pixel are stored contiguously.
                      2 says components are stored in component planes. Irrelevant for greyscale images.
        compression: compression level. defaults to no compression. More information can be found in the libtiff doc.
        tile_length: Only needed if method is "tile", sets the length of a tile. Must be a multiple of 16. Default: 256
        tile_width: Only needed if method is "tile", sets the width of a tile. Must be a multiple of 16. Default: 256

    Examples:
      >>> data = np.random.rand(100,100)
      >>> # data = np.random.randint(size=(100,100))
      >>> with pytiff.Tiff("example.tif", "w") as handle:
      >>>   handle.write(data, method="tile", tile_length=240, tile_width=240)
    """
    if self.file_mode not in ["w", "a", "w8", "a8"]:
      raise Exception("Write is only supported in .. write mode ..")

    cdef short photometric, planar_config, compression
    cdef short sample_format, nbits, samples_per_pixel

    photometric = options.get("photometric", MIN_IS_BLACK)
    if data.ndim == 3:
        photometric = RGB

    planar_config = options.get("planar_config", 1)
    compression = options.get("compression", NO_COMPRESSION)
    samples_per_pixel = 1
    if data.ndim == 3:
        samples_per_pixel = data.shape[2]
    sample_format, nbits = INVERSE_TYPE_MAP[data.dtype]

    ctiff.TIFFSetField(self.tiff_handle, tags.orientation, 1) # Image orientation , top left
    ctiff.TIFFSetField(self.tiff_handle, tags.samples_per_pixel, samples_per_pixel)
    ctiff.TIFFSetField(self.tiff_handle, tags.bits_per_sample, nbits)
    ctiff.TIFFSetField(self.tiff_handle, tags.image_length, data.shape[0])
    ctiff.TIFFSetField(self.tiff_handle, tags.image_width, data.shape[1])
    ctiff.TIFFSetField(self.tiff_handle, tags.sample_format, sample_format)
    ctiff.TIFFSetField(self.tiff_handle, tags.compression, compression) # compression, 1 == no compression
    ctiff.TIFFSetField(self.tiff_handle, tags.photometric, photometric) # photometric, minisblack
    ctiff.TIFFSetField(self.tiff_handle, tags.planar_configuration, planar_config) # planarconfig, contiguous not needed for gray
    self.logger.debug("Write config: {} bits per sample, {} samples per pixel, {} x {} image size".format(nbits,
        samples_per_pixel, data.shape[0], data.shape[1]))
    self.logger.debug("Type of input data: {}, max value: {} min value: {} C contiguous: {}".format(data.dtype, data.max(), data.min(), data.flags.c_contiguous))

    write_method = options.get("method", "tile")
    if write_method == "tile":
      self._write_tiles(data, **options)
    elif write_method == "scanline":
      self._write_scanline(data, **options)

    self._write_mode_n_pages += 1

  def _write_tiles(self, np.ndarray data, **options):
    cdef short tile_length, tile_width
    tile_length = options.get("tile_length", 240)
    tile_width = options.get("tile_width", 240)
    self.logger.debug("Writing tiles of size {} x {}".format(tile_length, tile_width))

    ctiff.TIFFSetField(self.tiff_handle, tags.tile_length, tile_length)
    ctiff.TIFFSetField(self.tiff_handle, tags.tile_width, tile_width)

    cdef np.ndarray buffer
    n_tile_rows = int(np.ceil(data.shape[0] / float(tile_length)))
    n_tile_cols = int(np.ceil(data.shape[1] / float(tile_width)))
    self.logger.debug("Number of tiles in a row: {}".format(n_tile_rows))
    self.logger.debug("Number of tiles in a column: {}".format(n_tile_cols))

    cdef unsigned int x, y
    for i in range(n_tile_rows):
      for j in range(n_tile_cols):
        y = i * tile_length
        x = j * tile_width
        buffer = data[y:(i+1)*tile_length, x:(j+1)*tile_width]
        to_pad = [(0, tile_length - buffer.shape[0]), (0, tile_width - buffer.shape[1])]
        if data.ndim ==3:
            to_pad += [(0,0)]
        buffer = np.pad(buffer, to_pad, "constant", constant_values=(0))
        self.logger.debug("Buffer array c contiguous: {}".format(buffer.flags.c_contiguous))

        ctiff.TIFFWriteTile(self.tiff_handle, <void *> buffer.data, x, y, 0, 0)

    ctiff.TIFFWriteDirectory(self.tiff_handle)

  def _write_scanline(self, np.ndarray data, **options):
    self.logger.debug("Writing scanlines")
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    self.logger.debug("Data array c contiguous: {}".format(data.flags.c_contiguous))

    cdef unsigned int rows_per_strip
    if "rows_per_strip" in options:#
      rows_per_strip = options["rows_per_strip"]
      ctiff.TIFFSetField(self.tiff_handle, tags.rows_per_strip, rows_per_strip)
    else:
      ctiff.TIFFSetField(self.tiff_handle, tags.rows_per_strip, ctiff.TIFFDefaultStripSize(self.tiff_handle, data.shape[1])) # rows per strip, use tiff function for estimate
    cdef np.ndarray row
    for i in range(data.shape[0]):
      row = data[i]
      ctiff.TIFFWriteScanline(self.tiff_handle, <void *>row.data, i, 0)
    ctiff.TIFFWriteDirectory(self.tiff_handle)

  def new_page(self, image_size, dtype, **options):
    """ adds a new page to the tiff file, and initializes chunk writing


    Args:
        image_size (array like (integer)): the size of the image
        dytpe (np.dtype): the dtype of the image
        photometric: determines how values are interpreted, either zero == black or zero == white.
                     MIN_IS_BLACK(default), MIN_IS_WHITE. more information can be found in the libtiff doc.
        planar_config: defaults to 1, component values for each pixel are stored contiguously.
                      2 says components are stored in component planes. Irrelevant for greyscale images.
        compression: compression level. defaults to no compression. More information can be found in the libtiff doc.
        tile_length: sets the length of a tile. Must be a multiple of 16. Default: 256
        tile_width: sets the width of a tile. Must be a multiple of 16. Default: 256
    """
    if self._unsaved_page:
        self.save_page()
    cdef short photometric, planar_config, compression
    cdef short sample_format, nbits
    cdef int length, width
    photometric = options.get("photometric", MIN_IS_BLACK)
    planar_config = options.get("planar_config", 1)
    compression = options.get("compression", NO_COMPRESSION)

    # cast to numpy.dtype. if this is not done, keys are not matching.
    self._dtype_write = np.dtype(dtype)
    dtype = np.dtype(dtype)
    sample_format, nbits = INVERSE_TYPE_MAP[dtype]
    length = image_size[0]
    width = image_size[1]
    self.image_length = image_size[0]
    self.image_width = image_size[1]

    cdef short tile_length, tile_width
    tile_length = options.get("tile_length", 256)
    tile_width = options.get("tile_width", 256)
    self.tile_length = tile_length
    self.tile_width = tile_width
    ctiff.TIFFSetField(self.tiff_handle, tags.tile_length, tile_length)
    ctiff.TIFFSetField(self.tiff_handle, tags.tile_width, tile_width)

    ctiff.TIFFSetField(self.tiff_handle, tags.orientation, 1) # Image orientation , top left
    ctiff.TIFFSetField(self.tiff_handle, tags.samples_per_pixel, 1)
    ctiff.TIFFSetField(self.tiff_handle, tags.bits_per_sample, nbits)
    ctiff.TIFFSetField(self.tiff_handle, tags.image_length, length)
    ctiff.TIFFSetField(self.tiff_handle, tags.image_width, width)
    ctiff.TIFFSetField(self.tiff_handle, tags.sample_format, sample_format)
    ctiff.TIFFSetField(self.tiff_handle, tags.compression, compression) # compression, 1 == no compression
    ctiff.TIFFSetField(self.tiff_handle, tags.photometric, photometric) # photometric, minisblack
    ctiff.TIFFSetField(self.tiff_handle, tags.planar_configuration, planar_config) # planarconfig, contiguous not needed for gray
    self._unsaved_page = True

  def __setitem__(self, key, item):
    """ enables chunkwise writing uses _chunk_writing """
    self.logger.debug("__setitem__ called")
    if not isinstance(key, tuple):
      if isinstance(key, slice):
        key = (key, slice(None,None,None))
      else:
        raise Exception("Only slicing is supported")
    elif len(key) < 3:
      key = key[0],key[1],0

    if not isinstance(key[0], slice) or not isinstance(key[1], slice):
      raise Exception("Only slicing is supported")

    x_range = np.array((key[1].start, key[1].stop))
    if x_range[0] is None:
      x_range[0] = 0
    if x_range[1] is None or x_range[1] > self.image_width:
      x_range[1] = self.image_width

    y_range = np.array((key[0].start, key[0].stop))
    if y_range[0] is None:
      y_range[0] = 0
    if y_range[1] is None or y_range[1] > self.image_length:
      y_range[1] = self.image_length

    shape = y_range[1] - y_range[0], x_range[1] - x_range[0]
    if shape != item.shape:
      raise ValueError("data shape :{} is not matching to the slice: {}".format(item.shape, shape))
    if self._dtype_write != np.dtype(item.dtype):
        raise ValueError("data dtype :{} is not matching to the image dtype: {}".format(item.dtype, self.dtype))
    self._write_chunk(item, x_pos=x_range[0], y_pos=y_range[0])

  def _write_chunk(self, np.ndarray data, **options):
    """ writes a chunk at the given position

    Args:
        data (np.ndarray): the chunk of the image
        x_pos, y_pos (integer): sets the postiton where the chunk is written Default:0
    """

    x_chunk = options.get("x_pos",0)
    y_chunk = options.get("y_pos",0)

    cdef unsigned int tile_length, tile_width
    tile_length = self.tile_length
    tile_width = self.tile_width


    cdef np.ndarray buffer
    n_tile_rows = int(np.ceil(data.shape[0] / float(tile_length)))
    n_tile_cols = int(np.ceil(data.shape[1] / float(tile_width)))

    dtype = data.dtype
    cdef unsigned int x, y
    for i in range(n_tile_rows):
      for j in range(n_tile_cols):
        y = i * tile_length
        x = j * tile_width
        buffer = data[y:(i+1)*tile_length, x:(j+1)*tile_width]
        buffer.astype(dtype)
        buffer = np.pad(buffer, ((0, tile_length - buffer.shape[0]), (0, tile_width - buffer.shape[1])), "constant", constant_values=(0))

        ctiff.TIFFWriteTile(self.tiff_handle, <void *> buffer.data, x_chunk+x, y_chunk+y, 0, 0)

  def read_tags(self):
    """  reads standard tags and saves them in a dictionary

        Returns
            the tags (dictionary) (they are also saved as an attribute of the pyTiff Object)
    """
    if self.file_mode != "r":
        raise Exception("Tag reading is only supported in read mode")
    _tags = {}
    for key in TIFF_TAGS:
        attribute_name, default_value, data_type, count = TIFF_TAGS[key]

        # if tiled dont read strip offsets and counts
        # if not tiled dont read tile offsets and counts
        if attribute_name in ["strip_byte_counts", "strip_offsets"] and self.is_tiled():
            continue
        if attribute_name in ["tile_byte_counts", "tile_offsets"] and not self.is_tiled():
            continue

        # if no string and count is None get variable length
        if count is None and data_type != 2:
          count = self._value_count(key)
          if count is None:
              self.logger.warn("Tag: {} not supported and omitted".format(attribute_name))
              continue
        self.logger.debug("name: {}, count: {}, data_type: {}".format(attribute_name, count, data_type))
        value, error_code = self._read_tag(key, data_type, count)
        if error_code == 1:
          self.logger.debug("Tag {} read!".format(attribute_name))
          if attribute_name == "bits_per_sample":
              self.logger.debug("convert bits per sample to an array of length samples per pixel")
              value = np.ones(self.samples_per_pixel, dtype=np.uint16) * value[0]
          _tags[tags[attribute_name]] = copy.deepcopy(value)

    self.tags = TagDict()
    self.tags.update(_tags)
    return self.tags

  def _read_tag(self, tag, data_type, count):
    """ reads a single attribute from a Tiff File

        Args:
            tag (integer): the attribute tag
            data_type (integer): the key for the numpy data type specified in TIFF_DATA_TYPES
            count (integer): number of elements for the tag
        Returns:
            tuple: the attribute (either a string or a numpy array of length count), error code (1 == success)
    """
    cdef np.ndarray data
    cdef void* d
    cdef unsigned short page, n_pages
    # if no data type, don't try to read
    if data_type is None:
        return None, 0

    # special case for strings
    if data_type == 2:
        self.logger.debug("string tag")
        return self._read_ascii(tag)

    # double count for RATIONAL datatype
    # every uint64 value has 2 uint32 values
    if data_type == 5:
        self.logger.debug("rational tag")
        data_type = np.dtype("uint32")
        count *= 2
    # the same goes for SRATIONAL
    elif data_type == 10:
        self.logger.debug("srational tag")
        data_type = np.dtype("int32")
        count *= 2
    # if neither RATIONAL nor SRATIONAL
    # use the mapped data type
    else:
        self.logger.debug("normal tag")
        data_type = TIFF_DATA_TYPES[data_type]

    # another special case is the page number
    # TIFFGetField expects two shorts and not a pointer
    if tag == tags.page_number:
        data = np.zeros(count, dtype=data_type)
        err = ctiff.TIFFGetField(self.tiff_handle, tag, &page , &n_pages)
        data[0] = page
        data[1] = n_pages
    # handle tags with count > 1, this only works if TIFFGetField expects a
    # pointer to an array. A buffer variable needs to be used because
    # TIFFGetField allocates the necessary memory itself. Afterwards the data
    # is saved in a numpy array.
    elif count > 1:
        err = ctiff.TIFFGetField(self.tiff_handle, tag, &d)
        if err == 1:
            data = _to_view(d, data_type, size=count)
        else:
            data = None
    # simple case if count == 1. Use a reference to allocated memory.
    else:
        data = np.zeros(count, dtype=data_type)
        err = ctiff.TIFFGetField(self.tiff_handle, tag, <void *> data.data)
    return data, err

  def _read_ascii(self, tag):
    """ reads an ascii bytestring from a Tiff File

        Args:
            tag (integer): the attribute tag

        Returns:
            the attribute (bytestring)
    """
    cdef char* desc = ''
    err = ctiff.TIFFGetField(self.tiff_handle, tag, &desc)
    str = <bytes>desc
    if str == "":
      str = None
    return str, err

  def set_tags(self, tagdict=None, **kwargs):
    """ writes the tag/value pairs in the dict to the Tiff File

        Tags must be written before image data is written to the current page.

        Args:
            kwargs (dictionary): consists of tag/value pairs, where
                tag (integer/string): either a tag or an attribute name
                value: the value which should be written to the Tiff File

        Examples:
            >>> tiff_file.set_tags(artist="John Doe"})
            >>> tiff_file.write(array)
    """
    if self.file_mode == "r":
        raise Exception("Tag writing is not supported in read mode")
    kwargs.update(tagdict)

    for _key in kwargs:
      if isinstance(_key, str):
          key = tags[_key]
      else:
          key = _key

      if key in TIFF_TAGS_NOT_WRITABLE:
          continue
      self._set_tag(key, kwargs[key])

  def _set_tag(self, tag, value):
    """  sets one tag in the tiff file


        Args:
            tag (integer/string): either a tag or attribute name
            value: the value which should be written to the Tiff File
    """

    cdef np.ndarray data
    cdef char* char_ptr
    cdef unsigned int count
    # helper variables
    cdef unsigned char page, total_pages
    cdef float fval
    if type(tag) == int:
      tag = tags(tag)
    elif isinstance(tag, str):
        tag = tags[tag]

    # special case for page number, same as for reading the tag
    if tags.page_number == tag:
      page = value[0]
      total_pages = value[1]
      err = ctiff.TIFFSetField(self.tiff_handle, tag, page, total_pages)
      return
    if isinstance(value, str):
      value = value + "\0"
      if PY3:
         py_byte_string = value.encode()
         char_ptr = py_byte_string
      else:
         char_ptr = value
      err = ctiff.TIFFSetField(self.tiff_handle, tag, char_ptr)
      return
    else:
      data = value

    assert isinstance(data, np.ndarray)
    if len(data) > 1:
        count = len(data)
        ctiff.TIFFSetField(self.tiff_handle, tag, count,<void *> data.data)
    else:
        if data.dtype == np.dtype("float32"):
            fval = data.item(0)
            err = ctiff.TIFFSetField(self.tiff_handle, tag, <float> data.item(0))
        err = ctiff.TIFFSetField(self.tiff_handle, tag, data.data[0])

  def save_page(self):
    """ saves the page """
    if self._unsaved_page:
        self._unsaved_page = False
        ctiff.TIFFWriteDirectory(self.tiff_handle)
        self._write_mode_n_pages += 1

  cdef _read_tile(self, unsigned int y, unsigned int x):
    cdef np.ndarray buffer = np.zeros((self.tile_length, self.tile_width, self.n_samples),dtype=self.dtype).squeeze()
    cdef ctiff.tsize_t bytes = ctiff.TIFFReadTile(self.tiff_handle, <void *>buffer.data, x, y, 0, 0)
    if bytes == -1:
      raise NotTiledError("Tiled reading not possible")
    return buffer

  def _value_count(self, tag):
    cdef short planarconfig
    ctiff.TIFFGetField(self.tiff_handle, tags.planar_configuration, &planarconfig)
    pool_samples_per_pixel = [
            TIFF_TAGS_REVERSE["bits_per_sample"],
            TIFF_TAGS_REVERSE["min_sample_value"],
            TIFF_TAGS_REVERSE["max_sample_value"],
            TIFF_TAGS_REVERSE["smin_sample_value"],
            TIFF_TAGS_REVERSE["smax_sample_value"],
            TIFF_TAGS_REVERSE["sample_format"]
            ]
    if tag in pool_samples_per_pixel:
        sys.stdout.flush()
        return self.samples_per_pixel
    elif tag == TIFF_TAGS_REVERSE["strip_offsets"] or tag == TIFF_TAGS_REVERSE["strip_byte_counts"]:
        if planarconfig == 1:
            return ctiff.TIFFNumberOfStrips(self.tiff_handle)
        if planarconfig == 2:
            return ctiff.TIFFNumberOfStrips(self.tiff_handle) * self.samples_per_pixel
    elif tag == TIFF_TAGS_REVERSE["tile_offsets"] or tag == TIFF_TAGS_REVERSE["tile_byte_counts"]:
        if planarconfig == 1:
            return ctiff.TIFFNumberOfTiles(self.tiff_handle)
        if planarconfig == 2:
            return ctiff.TIFFNumberOfTiles(self.tiff_handle) * self.samples_per_pixel
    elif tag == TIFF_TAGS_REVERSE["extra_samples"]:
        return self.extra_samples.size
    else:
        return None

  @property
  def pages(self):
    if self._pages is None:
      self._pages = []
      current = 0
      mode = "r"
      # use bigtiff if original file is opened as bigtiff
      if "8" in self.file_mode:
          mode += "8"
      while current < self.number_of_pages:
          page = Tiff(self.filename, mode)
          page.set_page(current)
          current += 1
          page._singlepage = True
          self._pages.append(page)
    return self._pages

class TagDict(dict):
    def __init__(self, *args, **kwargs):
        super(TagDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        res = super(TagDict, self).get(key, None)
        try:
            length = len(res)
        except:
            length = 2
        if length == 1:
            res = res[0]

        return res

