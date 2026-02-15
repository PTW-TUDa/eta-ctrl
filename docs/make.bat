@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)

if "%SPHINXAPIDOC%" == "" (
	set SPHINXAPIDOC=sphinx-apidoc
)

set SOURCEDIR=.
set BUILDDIR=_build
set STUBSDIR=_stubs
set MODULEDIR=../eta_ctrl

REM strict mode - fail on warnings
set STRICT_OPTS=-W --keep-going

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

if "%1" == "clean" goto clean
if "%1" == "linkcheck" goto linkcheck
if "%1" == "ci-html" goto ci-html

REM Default Sphinx targets
%SPHINXAPIDOC% --tocfile api -e -M -o %STUBSDIR% %MODULEDIR% %O%
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:clean
echo Cleaning build directories...
if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
if exist %STUBSDIR% rmdir /s /q %STUBSDIR%
goto end

:linkcheck
echo Checking for broken links...
%SPHINXAPIDOC% --tocfile api -e -M -o %STUBSDIR% %MODULEDIR% %O%
%SPHINXBUILD% -M linkcheck %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:ci-html
%SPHINXAPIDOC% --tocfile api -e -M -o %STUBSDIR% %MODULEDIR% %O%
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %STRICT_OPTS% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
