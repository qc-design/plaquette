# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -j auto
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile apidoc

livehtml: apidoc
	sphinx-autobuild --port 8107 -b html $(SOURCEDIR) $(BUILDDIR)/html --watch ../src -j auto \
		--re-ignore '.*/_build/jupyter_execute/.*' \
		--re-ignore '.*/.ipynb_checkpoints/.*' \
		--re-ignore '.*/\.~[^/]*\.ipynb'

apidoc:
	sphinx-apidoc -t _templates/apidoc -T -M -f -d 1 -o apidoc ../src/plaquette

clean:
	rm -rf _build
	rm -rf _autosummary
	rm -rf apidoc

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile apidoc
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
