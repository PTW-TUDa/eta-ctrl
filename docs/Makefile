# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = .
BUILDDIR      = _build
STUBSDIR      = _stubs
MODULEDIR     = ../eta_ctrl

STRICT_OPTS = -W --keep-going

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean ci-html linkcheck

# Clean build directories
clean:
	@echo "Cleaning build directories..."
	@rm -rf $(BUILDDIR) $(STUBSDIR)

# Explicit linkcheck target
linkcheck:
	@echo "Checking for broken links..."
	@$(SPHINXAPIDOC) --tocfile api -e -M -o $(STUBSDIR) $(MODULEDIR) $(O)
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

ci-html:
	@$(SPHINXAPIDOC) --tocfile api -e -M -o $(STUBSDIR) $(MODULEDIR) $(O)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(STRICT_OPTS) $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXAPIDOC) --tocfile api -e -M -o $(STUBSDIR) $(MODULEDIR) $(O)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
