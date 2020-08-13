Displacement Angle and Magnitude Measurements for Beating Cardiomyocytes
========================================================================

This package provides the ``calc_optical_flow.py`` script to analyze the motion
of phase movies of cardiomyocytes grown on different substrates and measure
the direction and magnitude of motion [1]_.

To analyze the example movies provided in the ``data`` folder in this package:

.. code-block:: bash

    $ calc_optical_flow.py ./data -o ./output

Additional options are available using the online help:

.. code-block:: bash

    $ calc_optical_flow.py -h

If you find this package useful, please cite:

.. [1] Kauss, M. A. et al. Cardiac Cell-Derived Matrices Impart Age-Specific Functional Properties to Human Cardiomyocytes. http://biorxiv.org/lookup/doi/10.1101/2020.07.31.231480 (2020) doi:10.1101/2020.07.31.231480.

Individual Modules
------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   flow
   utils

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
