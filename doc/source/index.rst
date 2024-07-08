.. dgmrf documentation master file, created by
   sphinx-quickstart on Wed Jan 17 15:31:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dgmrf's documentation!
=================================

Changelog:

* v0.2.1:

    - Add large graph support with power series and other modifications

* v0.2.0:

    - Consistent results between conv-DGMRF and graph-DGMRF. All formula must be correct now.

* v0.1.1:

    - Major code improvements for graph-DGMRF. We now have posterior means from exact and approximate computations.

* v0.1.0:

    - Major improvements and code cleaning. Satisfying results for the graph DGMRF now (no more stop_gradient, now using a BCOO A matrix that speed up a lot computations). All notebooks have been updated.

* v0.0.1:

    - Initial release

.. toctree::
   :maxdepth: 5
   :caption: Contents:

   dgmrf.rst

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Code examples

   notebooks/*


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
