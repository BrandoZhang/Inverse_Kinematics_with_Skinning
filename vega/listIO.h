/*************************************************************************
 *                                                                       *
 * Vega FEM Simulation Library Version 4.0                               *
 *                                                                       *
 * "listIO" library , Copyright (C) 2007 CMU, 2009 MIT                   *
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

#ifndef _LISTIO_H_
#define _LISTIO_H_

/*
  A class to load an integer list from a text file into memory. 
*/

#include <stdio.h>
#include <vector>
#include <set>
#include <istream>
#include <ostream>
#include <algorithm>

class ListIO
{
public:

  // returns 0 on success and non-zero otherwise
  // load: sort the loaded entries if sortAfterLoad is true
  // numbers are separated by ' ' or ','. Support syntax "1:5" to load {1,2,3,4,5}
  // comments start with '#'
  static int load(const char * filename, int * numListEntries, int ** listEntries, int offset=0, bool sortAfterLoad = true);
  static int save(const char * filename, int numListEntries, const int * listEntries, int offset=0);

  static int load(const char * filename, std::vector<int> & listEntries, int offset=0, bool sortAfterLoad = true);
  static int save(const char * filename, const std::vector<int> & listEntries, int offset=0);

  static int load(const char * filename, std::set<int> & listEntries, int offset=0);
  static int save(const char * filename, const std::set<int> & listEntries, int offset=0);

  static int load(std::istream & i, std::vector<int> & listEntries, int offset=0, bool sortAfterLoad = true);
  static int save(std::ostream & o, int numListEntries, const int * listEntries, int offset=0);
  inline static int save(std::ostream & o, const std::vector<int> & list, int offset=0) { return save(o, list.size(), list.data(), offset); }

  static int loadBinary(const char * filename, int * numListEntries, int ** listEntries, int offset=0);
  static int loadBinary(FILE * fin, int * numListEntries, int ** listEntries, int offset=0);
  static int saveBinary(const char * filename, int numListEntries, int * listEntries, int offset=0);
  static int saveBinary(FILE * fout, int numListEntries, int * listEntries, int offset=0);

  // loads/saves multiple lists to one binary file
  static int loadBinaryMulti(const char * filename, int * numLists, int ** numListEntries, int *** listEntries, int offset=0);
  static int saveBinaryMulti(const char * filename, int numLists, int * numListEntries, int ** listEntries, int offset=0);

  static void sort(int numListEntries, int * listEntries) { return std::sort(listEntries, listEntries + numListEntries); }
  static void print(int numListEntries, const int * listEntries);
  inline static void print(const std::vector<int> & list) { print(list.size(), list.data()); }

  static void stripBlanks(char * s);

protected:
};

#endif

