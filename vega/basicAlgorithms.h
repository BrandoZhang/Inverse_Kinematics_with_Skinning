/*************************************************************************
 *                                                                       *
 * Vega FEM Simulation Library Version 4.0                               *
 *                                                                       *
 * "basicAlgorithms" library , Copyright (C) 2018 USC                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * Code authors: Yijing Li, Jernej Barbic                                *
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

#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>
#include <cmath>
#include <map>

template<class Container>
int sizei(const Container & c) { return static_cast<int>(c.size()); }

// return true if all elements in [first, last] == value
template<class InputIt, class T>
bool allOf(InputIt first, InputIt last, const T & value)
{
  auto l = [&](const typename std::iterator_traits<InputIt>::value_type & it)
  {
    return it == value;
  };
  return std::find_if_not(first, last, l) == last;
}

template<class Container, class Range>
void insertRange(Container & a, const Range & b) { a.insert(b.begin(), b.end()); }

template<class Container, class Range>
void insertRangeToEnd(Container & a, const Range & b) { a.insert(a.end(), b.begin(), b.end()); }

// remove duplicate elements in vec and sort them as a side effect
// elements of type T need a default constructor
template<class T>
void sortAndDeduplicate(std::vector<T> & vec)
{
  std::sort(vec.begin(), vec.end());
  auto newEnd = std::unique(vec.begin(), vec.end());
  vec.resize(std::distance(vec.begin(), newEnd));
}

// remove duplicate elements in vec and sort them as a side effect
// use vector::erase to remove duplicate elements so that they don't need a default constructor
template<class T>
void sortAndDeduplicateWithErase(std::vector<T> & vec)
{
  std::sort(vec.begin(), vec.end());
  auto newEnd = std::unique(vec.begin(), vec.end());
  vec.erase(newEnd, vec.end());
}

// go through each index in range [first, last) and process the index by calling f(Index ID)
// different from std::for_each, which goes through an iterator range for(iter : range), and calls f(*iter)
template<class Index, class UnaryFunction>
void forEachIndex(Index first, Index last, UnaryFunction f)
{
  for(; first != last; ++first)
  {
    f(first);
  }
}

template<class Array>
auto convertArrayToFunction(const Array & arr) -> std::function<decltype(arr[0])(int)>
{
  return [&arr](int index) -> decltype(arr[index]) { return arr[index]; };
}

// run binary search on a range to find whehter a val is in the range
template<class RamdomAccessRange, class T>
bool binarySearchFound(const RamdomAccessRange & range, const T & val)
{
  return binary_search(range.begin(), range.end(), val);
}

template<class T>
T clamp(const T & a, const T & low, const T & high)
{
  if (a < low) { return low; }
  if (a > high) { return high; }
  return a;
}

template<class T>
void clampSelf(T & a, const T & low, const T & high)
{
  if (a < low) { a = low; }
  else if (a > high) { a = high; }
}

// return 1 if v > 0, -1 if v < 0, 0 if v == 0
template<class T>
int sign(const T & v)
{
  if (0 < v) return 1;
  if (v < 0) return -1;
  return 0;
}

// return 0.0 if v < 0.0
// useful when you compute a value which should be >= 0.0 but might gives negative values due to numerical errors
inline double sqrtSafe(double v)
{
  if (v < 0.0) { return 0.0; }
  return std::sqrt(v);
}

// for a range of elements [first, last) and a unary func on the element,
// find the element that gives max func(element)
template<class InputIt, class UnaryFunction>
InputIt maxFunctionValue(InputIt first, InputIt last, UnaryFunction func)
{
  if (first == last) { return last; }
  InputIt largest = first;
  auto value = func(*first);
  first++;
  for (; first != last; first++)
  {
    auto newValue = func(*first);
    if (value < newValue)
    {
      largest = first;
      value = newValue;
    }
  }
  return largest;
}

// compute median for [first, last)
// if first == last, return default value: value_type()
// computing median involves averaging two values, this is done here by dividing by value_type(2)
template<class RandomIt>
typename std::iterator_traits<RandomIt>::value_type median(RandomIt first, RandomIt last)
{
  typedef typename std::iterator_traits<RandomIt>::value_type value;
  if (first == last) { return value(); }
  auto size = distance(first, last);
  auto nth = first + size/2;
  nth_element(first, nth, last);
  if (size % 2 == 1) { return *nth; }

  // size % 2 == 0
  auto v0 = *nth;
  nth_element(first, nth-1, last);
  return (v0 + *(nth-1)) / value(2);
}

// remove elements from vector "inputVector" by given indices stored in "indices"
// input indices are sorted
template<class InputVector, class IndexRange>
void removeByIndices(InputVector & inputVector, const IndexRange & indices)
{
  int vecSize = inputVector.size();
//  int indSize = distance(indices.begin(), indices.end());
  auto ID = indices.begin();
  int newEnd = 0;
  for(int i = 0; i < vecSize; i++)
  {
    if (ID != indices.end() && i == *ID)
    {
      ID++;
      continue;
    }
    if (i != newEnd)
    {
      inputVector[newEnd] = std::move(inputVector[i]);
    }
    newEnd++;
  }
  inputVector.resize(newEnd);
}


// The template function must be run on a sorted range from first to last.
// It functions like std::unique, which operates on a sorted range and moves duplicated elements to the back of the range,
// and finally returns the iterator pointing to the end of the deduplicated elements.
// However, reduceDuplicates not only moves duplicate elements to the back, but also calls a reduce operator on
// those duplicated elements.
// reduce is a binary function: void reduce(T & entryA, T & entryB), reduces data in entryA and entryB and store the result into entryA
// e.g. input buffer is [0, 1, 2.1, 2.2, 3, 4.1, 4.2, 4.3], and we only compare with the digits before decimal point
// then the result is a buffer: [0, 1, 2.2, 3, 4.3, x, x, x], the function returns the iterator on the second 4,
// and reduce is called on (2.1,2.2) and (4.1,4.2), (4.1,4.3)
template<class ForwardIt, class BinaryReduce, class BinaryPredicate>
ForwardIt reduceDuplicates(ForwardIt first, ForwardIt last, BinaryReduce reduce, BinaryPredicate p)
{
  if (first == last)
    return last;

  ForwardIt result = first;
  while (++first != last)
  {
    if (p(*result, *first) == false) // if *result and *first are not equal
    {
      ++result; // move result forward
      if (result != first)           // if result and first does not point to the same location, then
        *result = std::move(*first); // move *first to *result
    }
    else // *result and *first are equal
    {
      reduce(*result, *first);
    }
  }
  return ++result;
}

template<class ForwardIt, class BinaryReduce>
ForwardIt reduceDuplicates(ForwardIt first, ForwardIt last, BinaryReduce reduce)
{
  using T = decltype(*first);
  return reduceDuplicates(first, last ,reduce, std::equal_to<const T &>());
}

#endif
