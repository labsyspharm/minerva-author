from libcpp.string cimport string

cdef extern from "tiffio.h":
  # structs
  cdef struct tiff:
    pass
  ctypedef struct TIFF:
    pass

  ctypedef struct TIFFField:
    pass

  # typedefs
  ctypedef unsigned int ttag_t
  ctypedef unsigned int ttile_t
  ctypedef int tsize_t
  ctypedef void* tdata_t
  ctypedef unsigned short tsample_t
  ctypedef unsigned short tdir_t
  ctypedef unsigned int ttile_t
  ctypedef unsigned int tstrip_t
  # functions
  # general functions
  int TIFFIsTiled(TIFF*)
  string TIFFGetVersion()
  const TIFFField* TIFFFieldWithTag(TIFF*, ttag_t)
  unsigned int TIFFFieldDataType(const TIFFField* )
  int TIFFGetField(TIFF*, ttag_t, ...)
  int TIFFSetField(TIFF* tif, ttag_t tag, ...)
  TIFF* TIFFOpen(const char*, const char*)
  void TIFFClose(TIFF*)
  # reading
  tsize_t TIFFReadTile(TIFF* tif, tdata_t buf, unsigned int x, unsigned int y, unsigned int z, tsample_t sample)
  int TIFFReadScanline(TIFF* tif, tdata_t buf, unsigned int row, tsample_t sample)
  # read helper
  ttile_t TIFFNumberOfTiles(TIFF* tif)
  tstrip_t TIFFNumberOfStrips(TIFF* tif)
  # write functions
  unsigned int TIFFDefaultStripSize(TIFF* tif, unsigned int estimate)
  int TIFFWriteScanline(TIFF* tif, tdata_t buf, unsigned int row, tsample_t sample)
  tsize_t TIFFWriteTile(TIFF* tif, tdata_t buf, unsigned int x, unsigned int y, unsigned int z, tsample_t sample)
  # directory functions
  tdir_t TIFFCurrentDirectory(TIFF* tif)
  int TIFFSetDirectory(TIFF* tif, tdir_t dir)
  int TIFFReadDirectory(TIFF* tif)
  int TIFFWriteDirectory(TIFF* tif)
  tdir_t TIFFNumberOfDirectories(TIFF* tiff)
  #RGBA functions
  int TIFFReadRGBAImage(TIFF* tif, unsigned int width, unsigned int height, unsigned int* raster, int stopOnError)
  int TIFFReadRGBATile(TIFF* tif, unsigned int x, unsigned int y, unsigned int* raster)
  unsigned short TIFFGetR(unsigned int pixel)
  unsigned short TIFFGetG(unsigned int pixel)
  unsigned short TIFFGetB(unsigned int pixel)
  unsigned short TIFFGetA(unsigned int pixel)
