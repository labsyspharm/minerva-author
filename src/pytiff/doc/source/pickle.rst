=================
Pickle a Tiff object
=================

While it is **not recommended**, a Tiff object can be pickled using the ``pickle`` or ``cPickle`` modules. It only saves the ``filename``, ``filemode``, ``bigtiff`` and the current ``page_number``.

When loading a Tiff object the saved information is used to create a new object.
Be aware that the same ``filemode`` is used. If you open a file in writing mode and pickle it, it will be opened in writing mode again. Thus overriding all written data.

Pickling is supported, so that a Tiff object can be used as a input source in a multiprocessing environment. Frameworks for multiprocessing often share data between different processes by sending pickled objects.
