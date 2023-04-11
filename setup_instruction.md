# Code setup instructions

This project requires three external libraries: Eigen, ADOL-C and our old friend OpenGL. Eigen and ADOL-C have been provided in the starter code.

## Eigen

[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. For this project, we will mostly use Eigen to solve linear systems. Eigen is (mostly) a header-only library, which means you don't need to compile and install it before using. For basic Eigen usage, see [Eigen: Getting started](https://eigen.tuxfamily.org/dox/GettingStarted.html). For using Eigen to solve linear systems, see the simple example file [EigenSolveExample.cpp](https://viterbi-web.usc.edu/~jbarbic/cs520-s23/assign3/EigenSolveExample.cpp) in the starter code.

## ADOL-C

[ADOL-C](https://github.com/coin-or/ADOL-C) is a library for computing first and higher derivatives of vector functions. For Windows users, we have provided pre-compiled ADOL-C headers, libs and dlls in the starter code. We have also provided a VS2017 project that has included the paths to ADOL-C. So no need to do anything. For Mac and Linux users, we have provided the source code and students can compile them using the following instructions:

First, we need to install some necessary tools for compiling ADOL-C.
For MacOS (tested on 10.14.2), we recommend installing Homebrew.
    Homebrew is a MacOS software package management system that provides an easy way to use libraries on MacOS as if using them on Linux.
    Then, open Terminal, run
    `$brew install autoconf automake libtool`
    to install the tools.
For Linux, run
    `$sudo apt-get install autoconf automake libtool`
    to install the tools.

Next, enter the ADOL-C folder: <starter code folder>/adolc/sourceCode/, run command

`$autoreconf -fi`

to create a configure script. If no errors are reported, run

`$./configure`

to create a Makefile. If no errors are reported, run

`$make`

to compile the code.

Finally run

`$make install`

to install ADOL-C at <your account's home folder>/adolc_base/.

If you want to install ADOL-C at a different location, or if you want to customize, you can read <starter code folder>/adolc/sourceCode/INSTALL for more information.
On Linux, you also need to add the path to ADOL-C libraries to LD_LIBRARY_PATH. 

For using ADOL-C, see https://core.ac.uk/download/pdf/62914383.pdf for a brief introduction.

We also provide a simple example file [ADOLCExample.cpp](https://viterbi-web.usc.edu/~jbarbic/cs520-s23/assign3/ADOLCExample.cpp) in the starter code which includes all the ADOL-C functions we need in this project.

## OpenGL

In this project, OpenGL is used for rendering.

For Windows and Linux users, no need to do anything for OpenGL.

For Mac users, it is a little bit tricky:

If you are using Mac OS X Mojave, make sure you update it to the latest version; otherwise, OpenGL errors can occur.
Next, use Homebrew to install freeglut, which is an implementation of GLUT:

`$brew install freeglut`

(Note that although macOS comes with a GLUT framework, it is now deprecated and may not be stable.)

Then, open Makefile, comment the line

`OPENGL_LIBS=-lGL -lGLU -lglut`

and uncomment

`OPENGL_LIBS=-framework OpenGL /usr/local/Cellar/freeglut/3.4.0/lib/libglut.dylib`

Now you should be able to compile the starter code in macOS.
