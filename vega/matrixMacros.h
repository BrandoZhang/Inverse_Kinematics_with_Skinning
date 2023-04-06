/*************************************************************************
 *                                                                       *
 * Vega FEM Simulation Library Version 4.0                               *
 *                                                                       *
 * "matrixIO" library , Copyright (C) 2007 CMU, 2009 MIT, 2018 USC       *
 * All rights reserved.                                                  *
 *                                                                       *
 * Code author: Jernej Barbic                                            *
 * http://www.jernejbarbic.com/vega                                      *
 *                                                                       *
 * Research: Jernej Barbic, Hongyi Xu, Yijing Li,                        *
 *           Danyong Zhao, Bohan Wang,                                   *
 *           Fun Shing Sin, Daniel Schroeder,                            *
 *           Doug L. James, Jovan Popovic                                *
 *                                                                       *
 * Funding: National Science Foundation, Link Foundation,                *
 *          Singapore-MIT GAMBIT Game Lab,                               *
 *          Zumberge Research and Innovation Fund at USC,                *
 *          Sloan Foundation, Okawa Foundation,                          *
 *          USC Annenberg Foundation                                     *
 *                                                                       *
 * This library is free software; you can redistribute it and/or         *
 * modify it under the terms of the BSD-style license that is            *
 * included with this library in the file LICENSE.txt                    *
 *                                                                       *
 * This library is distributed in the hope that it will be useful,       *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the file     *
 * LICENSE.TXT for more details.                                         *
 *                                                                       *
 *************************************************************************/

/*
  Commonly used macros in the matrix library.
*/

#ifndef _MATRIXMACROS_H_
#define _MATRIXMACROS_H_

/*
   Returns the array offset index of element located in row i and column j (both 0-indexed). The matrix has "numRows" rows, and is stored in LAPACK-style column-major order.
*/
#ifndef ELT
  #define ELT(numRows,i,j) (((long)j)*((long)numRows)+((long)i))
#endif

#ifndef MIN
  #define MIN(x,y) ((x)<(y) ? (x) : (y))
#endif

#ifndef MAX
  #define MAX(x,y) ((x)>(y) ? (x) : (y))
#endif


#endif
