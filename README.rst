``plaquette``: an all-encompassing fault-tolerance software package
===================================================================

.. image:: https://readthedocs.org/projects/plaquette-qec/badge/?version=latest
    :target: https://docs.plaquette.design/en/stable/?badge=stable
    :alt: Documentation Status

.. figure:: docs/index_code.png
    :align: right
    :alt: Distance-5 rotated planar code

``plaquette`` is a fully featured and easy-to-use library for studying quantum
error correction and fault tolerance.

The library features numerous quantum error correction codes, hardware-relevant
error models and versatile decoders that can be used together seamlessly,
and it is tailored to  accommodate both newcomers and experienced users alike!

**WARNING**: we are in the process of fully making this tool public, so expect
rough corners in the meantime! This is also why currently the project is
marked as "alpha" in the version specifier. Stay tuned. :)

Installation
------------

``plaquette`` is a pure Python package, so it can be easily installed via
``pip``::

   pip install plaquette --pre

``--pre`` is necessary while the only available release of ``plaquette`` is
marked as alpha. We will drop this requirement soon!


Feature Highlights
------------------

Our objective is to make plaquette a fully featured package that enables 
you to study quantum error correction and fault tolerance from start to 
finish. Some highlights:

- We’ve built into Plaquette the fastest decoding algorithm for finding 
  and correcting errors. The decoder is available here: `plaquette-unionfind <https://github.com/qc-design/plaquette-unionfind>`_.

- Plaquette allows running real-world experiments on quantum computers 
  from hardware providers. See `plaquette-ibm-backend <https://github.com/qc-design/plaquette-ibm-backend>`_ 
  for an example.

- Plaquette includes the broadest range of hardware errors that are amenable 
  to efficient simulations. Other than the usual Pauli errors, Plaquette 
  supports erasure errors and fabrication errors (stay tuned!), important 
  to several different hardware platforms. 

- Plaquette makes it effortless to design new error correction codes – 
  simply define the connectivity of the qubits, and plaquette takes care 
  of the rest making it easy to design new codes tailored to hardware 
  imperfections.

- Plaquette comes with extensive `documentation <https://docs.plaquette.design/>`_ 
  and comprehensive tutorials. 

An important aspect of plaquette is that these features have been developed 
concurrently, which reduces the need for glue code often required when 
interfacing many different components.

Documentation
-------------

Live pre-compiled documentation is available
`here <https://docs.plaquette.design/>`_.

Alternatively, the documentation can be built from the ``master`` branch by::

   make -C docs clean html

and the build files will be available in the ``docs/_build/html`` directory.

Need help? Want to contribute?
------------------------------

``plaquette`` is under heavy development, so it might have some rough corners that need
polishing. If you encounter something you think (or the docs say) should work but does
not, just open an `issue <https://github.com/qc-design/plaquette/issues/new>`_
or, if you also want to share a solution, a
`pull request <https://github.com/qc-design/plaquette/compare>`_! See
our `development standard <https://docs.plaquette.design/dev/index.html>`_ to
have an idea of how to match your suggestions to the codebase.

Want to simply share feedback or you're unsure how to do something? Open a new
`discussion <https://github.com/qc-design/plaquette/discussions/new/choose>`_!


Supporters
----------

``plaquette`` is developed and maintained by `QC Design <https://www.qc.design/>`_.
``plaquette`` is also supported by the German Ministry of Education and Research
(BMBF) via project `PhotonQ <https://www.photonq.de/>`_. An early prototype of
``plaquette`` was developed under the support of the BMBF project
`PhoQuant <https://www.quantentechnologien.de/forschung/foerderung/quantencomputer-demonstrationsaufbauten/phoquant.html>`_.


Try Plaquette today!
--------------------

Start exploring the challenges of fault tolerance with plaquette. We hope that 
you find it useful! To start your journey with plaquette, we recommend our 
`quickstart guide <https://docs.plaquette.design/en/latest/quickstart.html>`_, 
which will introduce you to many important aspects of the library.
