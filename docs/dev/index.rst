.. Copyright 2023, It'sQ GmbH and the plaquette contributors
   SPDX-License-Identifier: Apache-2.0

.. _devstd:

Development standard
====================

This document highlights the conventions and workflows used in all of
``plaquette`` codebase. This is a living document because technology and
software development move forward and we should not stick to processes just
because "We've always done it like this".

Setting up your environment
---------------------------

We support a minimum Python version of 3.10, so you should set up a virtual
environment with such an interpreter. Please use a "vanilla" Python environment
and not a ``conda`` one: if we ensure that we are "pip-compatible" then by
default we are ``conda``-compatible, but the opposite is not always true.

Most of the conventions described here should not burden you when writing code
but to keep a minimum of consistency each PR is checked against some rules
automatically. All of these checks on PR can be run locally by using
`pre-commit <https://pre-commit.com>`_. The following lines of shell code
should be enough to get you started!

.. code-block:: shell

   python3.10 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt -r requirements_dev.txt
   pre-commit install

Now whenever you commit some code, those changes will be checked against (most
of) the rules described here.

.. important::

   To avoid slowing down people who like to do frequent commits, **tests are
   not run as part of these local checks** but each PR will run tests on the
   proposed changes. You should manually run the tests just before pushing
   commits to a PR branch to avoid surprises.

Using ``pre-commit`` is highly encouraged, as it will ensure that most problems
are caught early and you won't forget running Tool X before pushing a PR. :)

Code development
----------------

Development loosely follows the tried and tested ``git-flow`` process of
collaboration. We simply do not make distinction between ``master``,
``release`` and ``development`` branches, we only use ``master``. When a good
chunk of new fixes and features land in ``master``, we tag and release
(using SemVer as good as possible).

The workflow is otherwise extremely simple:

#. create new branch, start working;
#. open PR with the features you want to propose;
#. wait for code review and address comments, if any;
#. if everything is OK, someone will squash and rebase into master.

Each PR should address **one** feature or fix at a time. We favour short-lived
branches to avoid conflicts as much as possible. Use descriptive commit
messages when pushing new features, and don't be shy of using the commit
message *body*.

Using the
`Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/>`_
convention is encouraged, but not mandatory. Since PRs should be short and to
the point, squashing everything should not result in any loss of information.

.. hint::

   If the feature you want to propose is substantial, it's best if you
   `discuss <https://github.com/qc-design/plaquette/discussions>`_ it
   beforehand, to align your goals with other work going on in ``plaquette``!


Naming style
~~~~~~~~~~~~

Please prefer longer but descriptive variable names and avoid Microsoft SDK
level madness like ``LPCWSTR`` when you actually want to say "Long Pointer to
Constant Wide String". For the rest, you should keep as close as it makes
sense to standard Python naming conventions, i.e.:

* class names are ``TitleCased``;
* variables/methods/functions are ``snake_cased``;
* module-level "constants" are ``ALLCAPS``;
* a private API (of any type: class, variable, function, etc.) starts
  ``_with_an_underscore``;
* names that shadow built-in names (like ``type``) should have an underscore
  appended (``type_``) or better use something different if possible.


Style and formatting
~~~~~~~~~~~~~~~~~~~~

Code can be written in any form but should be converted to a "standard" form
using the `Black <https://github.com/psf/black>`_ code formatting tool. Black
should be run before committing anything.

Lines should have at most 88 characters and this also applies to comments.
88 characters per line is black's default. Make exceptions only if
really necessary and wrap the offending lines in the special markers
``# fmt: off ... # fmt: on``. If there's only one line, ``# noqa`` at the end
of the line should be enough.

For *docstrings*, try to stay within 79 characters. This is to help people
using the ``help`` python REPL function to be able to read comfortably, or
even people using Jupyter Notebooks "Inspector" feature.

Linters
+++++++

Code-style is in general enforced through a series of linters which have the
capability of fixing small annoyances for you. We use
`ruff <https://beta.ruff.rs/docs/>`_ for all linting because it implements most
well-known Python ecosystem linters like ``isort``, ``pycodestyle``, etc in one
single tool. You should run ``ruff`` after fixing your formatting with
``black``, they are made to be 100% compatible (with the settings included in
this repo).

Testing
-------

Since ``plaquette`` deals with fairly advanced scientific topics, we usually
run two types of tests, nicknamed "slow" and "not slow" (incidentally that is
the ``pytest`` marker used to discriminate between the two types).

The "slow" tests are checks against known published results in the literature
against we test our own results obtained with ``plaquette``. These tests take
long to run, in the order of 10-15 minutes, and are used both to ensure
correctness (as much as possible) and to avoid regressions.

The "not slow" tests are more conventional software-engineering tests, like
unit and integration tests, which ensure that the codebase works. This tells
you nothing about the *correctness* of whatever the code produces, only that
there are no obvious bugs.

Unit tests are performed using `PyTest <https://docs.pytest.org/>`_. They can
be run as follows:

.. code-block:: console

   $ cd plaquette
   $ pytest
   ========================== test session starts ===========================
   ...
   src/plaquette/somemod/__init__.py ....                              [ 25%]
   src/plaquette/othermod/submod.py ...                                [ 50%]
   tests/some_tests.py ...                                             [100%]

   =========================== 4 passed in 0.16s ============================

Unit tests
~~~~~~~~~~

Any new feature should come with its own unit test. If the feature is simple
enough, a ``doctest`` will suffice, no need to start exploring its whole
exponentially-expanding parameter space. For more complicated tests that
require some initial set-up, a separate test case in the ``tests`` folder is
necessary.


Static type analysis
~~~~~~~~~~~~~~~~~~~~

We use `MyPy <http://mypy-lang.org/>`_ static type checking. Since most of the
codebase is type-hinted, this should catch most bugs before running any line of
code or test, but of course it's not a substitute for proper testing.

.. note::

   If you use VSCode, and by default you rely on PyRight for type-checking,
   some checks differ *by design* between MyPy and PyRight.

Writing docs
------------

Sphinx is used for documenting code in docstrings and for writing the main
documentation pages. Sphinx uses restructured text (reST) as markup format.
Here's some links to get you started with it:

* `Basic formatting <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
* `Syntax for tables (list tables) <https://docutils.sourceforge.io/docs/ref/rst/directives.html#list-table>`_
* `Docstring example <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html>`_ and
  `Docstring reference <https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain>`_
* `More examples <https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion>`_

Docstrings
~~~~~~~~~~

Docstrings in general follow `PEP 257 <https://peps.python.org/pep-0257/>`_
guidelines. In particular:

DO
   * use triple double-quotes (``"""``);
   * put the closing quotes in a line by themselves;
   * align body indentation with the opening triple quotes.

DON'T
   * copy the signature of the method/function in the docstring;
   * start the summary on the line after the opening triple quotes.

Docstring style
~~~~~~~~~~~~~~~

Docstrings follow the "Google"
`convention <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_
for their structure **with one exception**: class attributes are documented
directly under their declaration::

   class Example:
       """A useful class."""

       def __init__(self):
           self.foo: int = 1
           """An important parameter."""


.. important:: **DO NOT** use Sphinx "comment-like" syntax (``#: An important
   parameter``) since it will not show up in editors or IDEs documentation
   tooltips.

Documenting types
~~~~~~~~~~~~~~~~~

Do not write type information inside arguments docstrings::

   def f(x):
       """Compute the square of x

       Args:
         x (float): the number to square.

       Returns:
         the square of ``x``.
       """
       return x * x

Use type annotations instead::

   def f(x: float) -> float:
       """Compute the square of x

       Args:
         x: Value to be squared.

       Returns:
         The square of ``x``.
       """
       return x * x

Sphinx already provides types in the function/method signature when rendering
the docs, so reiterating this information in the parameter description is
pointless.

Example snippets
~~~~~~~~~~~~~~~~

Wherever possible add a ``doctest`` example to your docstrings, which uses
the feature you are documenting in the simplest possible way which still makes
sense.

The ``Examples:`` section can have lines starting with ``>>> ``, which is
Python's REPL prompt. This marks the example as a ``doctest`` and this will
be *executed* during tests and the output of the prompt lines will be compared
with the text underneath. If they do not match, the test will fail.

**This catches a lot of bugs** and you should **always** include a ``doctest``
example wherever possible. Not only that, but it also catches cases where you
forgot to update the docstring after making some changes.

.. seealso::

   For more information you should check the actual
   `doctest <https://docs.python.org/3/library/doctest.html>`_ documentation.
