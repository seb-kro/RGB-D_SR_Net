#!/bin/bash
make clean
rm -r source/autodoc
sphinx-apidoc -e -o source/autodoc ../src
make html