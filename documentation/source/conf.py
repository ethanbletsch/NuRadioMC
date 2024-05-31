#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# NuRadio documentation build configuration file, created by
# sphinx-quickstart on Tue Apr 27 09:47:47 2021.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import time
import fnmatch
import NuRadioMC # we need this for the version number
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    # 'autoapi.extension',
    'numpydoc',
    'sphinx.ext.intersphinx',
    # 'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    # 'sphinx.ext.autosummary',
    # 'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['.templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'main'

# General information about the project.
project = 'NuRadio'
copyright = '{}, The NuRadio Group'.format(time.asctime()[-4:])
author = 'The NuRadio Group'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = NuRadioMC.__version__
# The full version, including alpha/beta/rc tags.
release = NuRadioMC.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 5
}
html_static_path = ['custom_scripts']
html_css_files = ['styling.css']

html_logo = 'logo_small.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['.static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'NuRadiodoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'NuRadio.tex', 'NuRadio Documentation',
     'The NuRadio Group', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'nuradio', 'NuRadio Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'NuRadio', 'NuRadio Documentation',
     author, 'NuRadio', 'One line description of project.',
     'Miscellaneous'),
]



# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']



# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'numpy': ("https://numpy.org/doc/stable", None)
}
default_role = 'autolink' #TODO: probably switch to py:obj?

# Make sure the target is unique
autosectionlabel_prefix_document = True

# Don't make toctrees for class methods (doesn't seem to work with apidoc)
numpydoc_class_members_toctree = False

autoclass_content = 'both' # include __init__ docstrings in class description
autodoc_default_options = {
    'show-inheritance': True, # show 'Bases: Parent' for classes that inherit from parent classes
    'inherited-members': True, # also document inherited methods; mostly done to avoid missing cross-references
}
autodoc_member_order = 'bysource' # list methods/variables etc. by the order they are defined, rather than alphabetically
# 
# coverage_ignore_modules


autodoc_mock_imports = [
    'ROOT', 'mysql-python', 'pygdsm', 'MySQLdb', 'healpy', 'scripts',
    'uproot', 'radiopropa', 'plotly', 'past',
    'nifty5', 'mattak', 'MCEq', 'crflux'
    ]
# Raise warnings if any cross-references are broken
nitpicky = True

# this ignores some cross-reference errors inside docstrings
# that we don't care about
nitpick_ignore_regex = [
    ("py:class", "aenum._enum.Enum"),
    ("py:class", "aenum.Enum"),
    ("py:class", "tinydb_serialization.Serializer"),
    ("py:class", "radiopropa.ScalarField"),
    ("py:class", "nifty5.*"),
    ("py:obj",".*__call__") # not sure why this method is listed sometimes - it shouldn't be?
]

# def skip_modules(app, what, name, obj, skip, options):
#     if skip: # we ignore anything autodoc is configured to ignore
#         return skip
#     exclusions = [
#         '[/\]setup', '[ET][0-9][0-9]*'
#     ]
#     # print(name, what)
#     return any([fnmatch.fnmatch(name, pat) for pat in exclusions])

# def setup(app):
#     app.connect('autodoc-skip-member', skip_modules)
