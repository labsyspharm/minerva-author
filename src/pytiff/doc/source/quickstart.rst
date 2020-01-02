=================
Quick start guide
=================

This document talks you through everything you need to get started with pytiff.

----------
Installing
----------

Clone the github repository and install pytiff using setup.py or pip.
Requirements are listed in the requirements.tx:

- numpy
- cython
- libtiff C library > 4.0

Using pip:

.. code:: bash

  pip install pytiff

Install from sources:

.. code:: bash

  git clone https://github.com/FZJ-INM1-BDA/pytiff.git
  cd pytiff
  pip install .

You can also use the -e option with pip for development purposes. Be aware
that you have to rebuild the cython parts if they are changed.

.. code:: bash

  python setup.py build_ext --inplace

-------------------
Reading a tiff file
-------------------

Pytiff can read greyscale as well as RGB(A) images. For a greyscale image it assumes a tiled image in order
to be able to load parts of it. If this is not the case the whole image is loaded and cropped afterwards.
The following code returns a numpy array with the shape (100, 200).

.. code:: python

  import pytiff

  with pytiff.Tiff("test_data/small_example_tiled.tif") as handle:
    part = handle[100:200, 200:400]

-----------------------------
Reading a multipage tiff file
-----------------------------

A multipage tiff file can be read by iterating over the pages. The `pages` attribute returns a list of pages:

.. code:: python

  import pytiff

  with pytiff.Tiff("test_data/multi_page.tif") as handle:
    for page in handle.pages:
      print("Current shape: {}".format(handle.shape))
      current_page = handle[:]

or manually:

.. code:: python

  import pytiff

  with pytiff.Tiff("test_data/multi_page.tif") as handle:
    for p in range(handle.number_of_pages):
      handle.set_page(p)
      print("Current shape: {}".format(handle.shape))
      current_page = handle[:]

-------------------
Writing a tiff file
-------------------

Pytiff is able to save two dimensional numpy arrays as images.

.. code:: python

  import numpy as np
  import pytiff
  with pytiff.Tiff("test_data/tmp.tif", "w") as handle:
    data = np.random.randint(low=0, high=255, size=(500, 300), dtype=np.uint8)
    handle.write(data, method="scanline")

-----------------------------
Writing a multipage tiff file
-----------------------------

A multipage tiff file can be created by calling the write method multiple times.
The following code creates a tiff file with 5 pages.

.. code:: python

  import numpy as np
  import pytiff
  with pytiff.Tiff("test_data/tmp.tif", "w") as handle:
    for i in range(5):
      data = np.random.randint(low=0, high=255, size=(100, 100), dtype=np.uint8)
      handle.write(data, method="tile")

----------------------
Writing a bigtiff file
----------------------

A bigtiff file can be created by telling pytiff to use the bigtiff mode.

.. code:: python

  import numpy as np
  import pytiff
  with pytiff.Tiff("test_data/tmp.tif", "w", bigtiff=True) as handle:
    data = np.random.randint(high=255, size=(80000, 80000), dtype=np.uint8)
    handle.write(data, method="tile")

---------------------------------
Reading and Writing Standard Tags
---------------------------------

A tiff file can contain many tags. Pytiff supports reading and writing baseline tags and some more.

.. code:: python

    import pytiff
    with pytiff.Tiff("test_data/small_example.tif") as handle:
      for k in handle.tags:
        print("{key}: {value}".format(key=k, value=tags[k]))

Writing tags has to be done before image data is written to the current page.

.. code:: python

    import pytiff
    with pytiff.Tiff("test_data/tmp.tif", "w") as handle:
      handle.set_tags(image_description="Image description")
      handle.write(data)

All available tags can be found at `pytiff.tags`.

----------------
More information
----------------

More information on the available methods and attributes can be found in the api documentation.
