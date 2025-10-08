@echo Off

pushd %~dp0

rem Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" (
	goto help
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
echo.Please use `make ^<target^>` where ^<target^> is one of
echo.  html        to make standalone HTML files
echo.  dirhtml     to make HTML files named index.html in directories
echo.  singlehtml  to make a single large HTML file
echo.  pickle      to make pickle files
echo.  json        to make JSON files
echo.  epub        to make an epub
echo.  latex       to make LaTeX files, you can set PAPER=a4 or PAPER=letter
echo.  latexpdf    to make LaTeX files and run them through pdflatex
echo.  latexpdfja  to make LaTeX files and run them through platex/dvipdfmx
echo.  text        to make text files
echo.  man         to make manual pages
echo.  texinfo     to make Texinfo files
echo.  info        to make Texinfo files and run them through makeinfo
echo.  gettext     to make PO message catalogs
echo.  changes     to see what's changed since last build
echo.  xml         to make Docutils-native XML files
echo.  pseudoxml   to make pseudoxml-XML files for display purposes
echo.  linkcheck   to check all external links for integrity
echo.  doctest     to run all doctests embedded in the documentation (if enabled)
echo.  coverage    to run coverage check of the documentation (if enabled)
echo.  clean       to remove everything in the build directory

:end
popd
