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

#include "listIO.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
using namespace std;

int ListIO::load(istream & fin, std::vector<int> & listEntries, int offset, bool sortAfterLoad)
{
  listEntries.clear();

  int lastNum = -1;
  bool hasLastNum = false, consecutive = false;
  fin >> ws;
  string line;
  istringstream ss;
  while(!fin.eof())
  {
    line.clear();
    getline(fin, line);
    fin >> ws;
    if (line.size() == 0)
      continue;
    if (line[0] == '#')
      continue; // it's a comment

    ss.str(line);
    ss.seekg(0);
    while(!ss.eof())
    {
      int k = 0;
      ss.clear();
      ss >> k;
      if (!ss.fail())
      {
        if (consecutive) // we have loaded "a : b"
        {
          for(int i = lastNum+1; i < k; i++) 
            listEntries.push_back(i - offset);
          consecutive = false;
        }
        listEntries.push_back(k - offset);
        lastNum = k;
        hasLastNum = true;
      } 
      else
      {
        ss.clear(); //clear fail bit
        char separateSymbol = 0;
        ss >> separateSymbol;
        if (separateSymbol == ':' && hasLastNum)
          consecutive = true;
        else if (separateSymbol == ',') // including the case separateSymbol == ','
          consecutive = false;
        else // unknown symbols, do nothing
          consecutive = false; 
      }
      ss >> ws;
    } // end while (!ss.eof)
  } // end while (!fin.eof)

  // sort the list entries
  if (sortAfterLoad)
    std::sort(listEntries.begin(), listEntries.end());

  return 0;
}

int ListIO::load(const char * filename, std::vector<int> & listEntries, int offset, bool sort)
{
  ifstream fin(filename,ios::binary);
  if (!fin)
  {
    printf("Error: could not open file %s.\n",filename);
    return 1;
  }
  
  return load(fin, listEntries, offset, sort);
}

int ListIO::load(const char * filename, int * numListEntries, int ** listEntries, int offset, bool sort)
{ 
  *numListEntries = 0;
  *listEntries = NULL;

  vector<int> nums;
  int ret = load(filename, nums, offset, sort);
  if (ret != 0)
    return ret;

  *numListEntries = nums.size();
  *listEntries = (int*) malloc (sizeof(int) * nums.size());
  memcpy(*listEntries, nums.data(), sizeof(int) * nums.size());

  return 0;
}

int ListIO::save(const char * filename, const std::vector<int> & listEntries, int offset)
{
  return save(filename, listEntries.size(), listEntries.data(), offset);
}

int ListIO::load(const char * filename, std::set<int> & listEntries, int offset)
{
  vector<int> v;
  int ret = load(filename, v, offset, false);
  if (ret != 0)
    return ret;
  listEntries.clear();
  listEntries.insert(v.begin(), v.end());
  return 0;
}

int ListIO::save(const char * filename, const std::set<int> & listEntries, int offset)
{
  vector<int> v(listEntries.begin(), listEntries.end());
  return save(filename, v, offset);
}

int ListIO::save(std::ostream & o, int numListEntries, const int * listEntries, int offset)
{
  for(int nv=0; nv < numListEntries; nv++)
  {
    o << listEntries[nv] + offset;
    if (nv % 8 == 7)
      o << '\n';
  }
  o << '\n';

  return 0;
}
  
int ListIO::save(const char * filename, int numListEntries, const int * listEntries, int offset)
{
  // comma-separated text file of fixed vertices
  FILE * fout;
  fout = fopen(filename, "w");
  if (!fout)
  {
    printf("Error: could not open list file %s.\n",filename);
    return 1;
  }

  for(int nv=0; nv < numListEntries; nv++)
  {
    fprintf(fout, "%d,", listEntries[nv] + offset);
    if (nv % 8 == 7)
      fprintf(fout,"\n");
  }
  fprintf(fout,"\n");

  fclose(fout);

  return 0;
}

void ListIO::print(int numListEntries, const int * listEntries)
{
  for(int nv=0; nv < numListEntries; nv++)
  {
    printf("%d,", listEntries[nv]);
    if (nv % 8 == 7)
      printf("\n");
  }
  printf("\n"); fflush(NULL);
}

int ListIO::loadBinary(const char * filename, int * numListEntries, int ** listEntries, int offset)
{
  FILE * fin;
  fin = fopen(filename, "rb");
  if (!fin)
  {
    printf("Error: could not open list file %s.\n",filename);
    return 1;
  }

  int code = loadBinary(fin, numListEntries, listEntries, offset);
  fclose(fin);
  return code;
}

int ListIO::loadBinary(FILE * fin, int * numListEntries, int ** listEntries, int offset)
{
  int item = fread(numListEntries, sizeof(int), 1, fin);
  if (item != 1)
  {
    printf("Error: could not read the number of list entries.\n");
    return 1;
  }
  *listEntries = (int*) malloc (sizeof(int) * (*numListEntries));

  for(int nv=0; nv < *numListEntries; nv++)
  {
    item = fread(&(*listEntries)[nv], sizeof(int), 1, fin);
    if (item != 1)
    {
      printf("Error: could not read the list entry %d.\n", nv);
      return 1;
    }
  }

  for(int nv=0; nv < *numListEntries; nv++)
    (*listEntries)[nv] -= offset;

  return 0;
}

int ListIO::saveBinary(const char * filename, int numListEntries, int * listEntries, int offset)
{
  FILE * fout;
  fout = fopen(filename, "wb");
  if (!fout)
  {
    printf("Error: could not open list file %s.\n",filename);
    return 1;
  }

  int code = saveBinary(fout, numListEntries, listEntries, offset);
  fclose(fout);
  return code;
}

int ListIO::saveBinary(FILE * fout, int numListEntries, int * listEntries, int offset)
{
  fwrite(&numListEntries, sizeof(int), 1, fout);

  for(int nv=0; nv < numListEntries; nv++)
  {
    int value = listEntries[nv] + offset;
    fwrite(&value, sizeof(int), 1, fout);
  }

  return 0;
}

// loads/saves multiple lists to one binary file
int ListIO::loadBinaryMulti(const char * filename, int * numLists, int ** numListEntries, int *** listEntries, int offset)
{
  FILE * fin;
  fin = fopen(filename, "rb");
  if (!fin)
  {
    printf("Error: could not open list file %s.\n",filename);
    return 1;
  }

  int item = fread(numLists, sizeof(int), 1, fin);
  if (item != 1)
  {
    printf("Error: could not read the number of list.\n");
    return 1;
  }

  *numListEntries = (int*) malloc (sizeof(int) * *numLists);
  *listEntries = (int**) malloc (sizeof(int*) * *numLists);

  int code = 0;
  for(int i=0; i<*numLists; i++)
    code = code || loadBinary(fin, &((*numListEntries)[i]), &((*listEntries)[i]), offset);

  fclose(fin);

  return code;
}

int ListIO::saveBinaryMulti(const char * filename, int numLists, int * numListEntries, int ** listEntries, int offset)
{
  FILE * fout;
  fout = fopen(filename, "wb");
  if (!fout)
  {
    printf("Error: could not open list file %s.\n",filename);
    return 1;
  }

  fwrite(&numLists, sizeof(int), 1, fout);

  int code = 0;
  for(int i=0; i<numLists; i++)
    code = code || saveBinary(fout, numListEntries[i], listEntries[i], offset);

  fclose(fout);

  return code;
}

