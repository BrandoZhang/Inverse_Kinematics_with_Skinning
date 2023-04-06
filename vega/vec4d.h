/*
* Copyright (c) 2008, Carnegie Mellon University
* All rights reserved.
*
* Code author: Jernej Barbic
* Research: Jernej Barbic, Doug L. James
* Funding: NSF, Link Foundation
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Carnegie Mellon University, nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE* DISCLAIMED. IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  A simple class for vector algebra on 4D vectors 
  (summation, normalization, dot product, cross product, etc.).
*/

#ifndef _MINIVEC4D_H_
#define _MINIVEC4D_H_

#include <stdio.h>
#include <cmath>
#include <ostream>
#include <cstring>

class Vec4d
{
public:
  inline Vec4d() {}
  inline Vec4d(double x, double y, double z, double w) {elt[0]=x; elt[1]=y; elt[2]=z; elt[3]=w;}
  inline explicit Vec4d(double entry); // create a vector with all entries "entry" (can create zero vector for entry=0.0)
  inline Vec4d(const double vec[4]); // create a vector from the array of four doubles pointed to by "vec"
  inline Vec4d(const Vec4d & vec);

  inline void set(double x0, double x1, double x2, double x3); // assign vector [x0, x1, x2, x3]
  inline void set(double value); // set all elements to value

  inline Vec4d & operator=(const Vec4d & source);
  inline bool operator==(const Vec4d & vec2) const;
  inline bool operator!=(const Vec4d & vec2) const;

  inline const Vec4d operator+ (const Vec4d & vec2) const;
  inline Vec4d & operator+= (const Vec4d & vec2);

  inline const Vec4d operator- (const Vec4d & vec2) const;
  inline Vec4d & operator-= (const Vec4d & vec2);

  inline const Vec4d operator* (double scalar) const;
  inline Vec4d & operator*= (double scalar);

  inline Vec4d operator/ (double scalar) const;
  inline Vec4d & operator/= (double scalar);

  // operator for Vec4d to be used as a key in std::set, std::map, etc.
  inline bool operator < (const Vec4d & vec2) const;

  friend inline Vec4d operator* (double scalar, const Vec4d & vec2);
  friend inline Vec4d operator/ (double scalar, const Vec4d & vec2);
  friend inline Vec4d operator- (const Vec4d & vec1);

  friend inline double dot(const Vec4d & vec1, const Vec4d & vec2); // dot product

  friend inline Vec4d norm(const Vec4d & vec1); // returns normalized vector (unit length)
  inline void normalize(); // normalize itself without returning anything

  friend inline std::ostream &operator << (std::ostream &s, const Vec4d &v);
  void print() const;

  inline double & operator[] (int i) { return elt[i]; } // v[i] returns i-th entry of v
  inline const double & operator[] (int i) const { return elt[i]; }

  operator const double * () const { return &elt[0]; }

  const double * data() const { return &elt[0]; }
  double * data() { return &elt[0]; }

  // copies the vector into an array of length 4
  inline void convertToArray(double vecArray[4]) const;
  // adds the vector into an array of length 4
  inline void addToArray(double vecArray[4]) const;

  inline bool hasNaN() const;
  
protected:
  double elt[4];
};

// === below is the implementation ===

inline Vec4d::Vec4d(double entry)
{
  set(entry);
}

inline Vec4d::Vec4d(const double vec[4])
{
  memcpy(elt, vec, sizeof(double) * 4);
}

inline Vec4d::Vec4d(const Vec4d & vec)
{
  memcpy(elt, vec.elt, sizeof(double) * 4);
}

inline Vec4d & Vec4d::operator=(const Vec4d & source)
{
  memcpy(elt, source.elt, sizeof(double) * 4);
  return *this;
}

inline bool Vec4d::operator==(const Vec4d & vec2) const
{
  return ((elt[0] == vec2[0]) &&
          (elt[1] == vec2[1]) &&
          (elt[2] == vec2[2]) &&
          (elt[3] == vec2[3]));
}

inline bool Vec4d::operator!=(const Vec4d & vec2) const
{
  return ((elt[0] != vec2[0]) ||
          (elt[1] != vec2[1]) ||
          (elt[2] != vec2[2]) ||
          (elt[3] != vec2[3]));
}

inline bool Vec4d::operator<(const Vec4d & vec2) const
{
  if(elt[0] < vec2[0]) 
    return true;
  if(elt[0] > vec2[0]) 
    return false;
  if(elt[1] < vec2[1]) 
    return true;
  if(elt[1] > vec2[1]) 
    return false;
  if(elt[2] < vec2[2]) 
    return true;
  if(elt[2] > vec2[2]) 
    return false;
  return elt[3] < vec2[3];
}

inline Vec4d operator* (double scalar, const Vec4d & vec2)
{
  Vec4d result = vec2;
  result.elt[0] *= scalar;
  result.elt[1] *= scalar;
  result.elt[2] *= scalar;
  result.elt[3] *= scalar;

  return result;
}

inline Vec4d operator- (const Vec4d & vec1)
{
  return vec1 * (-1);
}

inline const Vec4d Vec4d::operator+ (const Vec4d & vec2) const
{
  Vec4d sum = *this;
  sum.elt[0] += vec2.elt[0];
  sum.elt[1] += vec2.elt[1];
  sum.elt[2] += vec2.elt[2];
  sum.elt[3] += vec2.elt[3];

  return sum;
}

inline Vec4d & Vec4d::operator+= (const Vec4d & vec2)
{
  elt[0] += vec2.elt[0];
  elt[1] += vec2.elt[1];
  elt[2] += vec2.elt[2];
  elt[3] += vec2.elt[3];

  return *this;
}

inline const Vec4d Vec4d::operator- (const Vec4d & vec2) const
{
  Vec4d sum = *this;
  sum.elt[0] -= vec2.elt[0];
  sum.elt[1] -= vec2.elt[1];
  sum.elt[2] -= vec2.elt[2];
  sum.elt[3] -= vec2.elt[3];

  return sum;
}

inline Vec4d & Vec4d::operator-= (const Vec4d & vec2)
{
  elt[0] -= vec2.elt[0];
  elt[1] -= vec2.elt[1];
  elt[2] -= vec2.elt[2];
  elt[3] -= vec2.elt[3];

  return *this;
}

inline double dot(const Vec4d & vec1, const Vec4d & vec2)
{
  return (vec1.elt[0] * vec2.elt[0] + vec1.elt[1] * vec2.elt[1] + vec1.elt[2] * vec2.elt[2] + vec1.elt[3] * vec2.elt[3]);
}

inline const Vec4d Vec4d::operator* (double scalar) const
{
  return (Vec4d(elt[0] * scalar, elt[1] * scalar, elt[2] * scalar, elt[3] * scalar));
}

inline Vec4d Vec4d::operator/ (double scalar) const
{
  return (Vec4d(elt[0] / scalar, elt[1] / scalar, elt[2] / scalar, elt[3] / scalar));
}

inline Vec4d & Vec4d::operator*= (double scalar)
{
  *this = (*this) * scalar;
  return *this;
}

inline Vec4d & Vec4d::operator/= (double scalar)
{
  *this = (*this) / scalar;
  return *this;
}

inline double len(const Vec4d & vec1)
{
  return(sqrt(dot(vec1,vec1)));
}

inline Vec4d norm(const Vec4d & vec1)
{
  return vec1 / len(vec1); 
}

inline double len2(const Vec4d & vec1)
{
  return(dot(vec1,vec1));
}

inline std::ostream &operator << (std::ostream & s, const Vec4d & v)
{
  return(s << '[' << v[0] << ' ' << v[1] << ' ' << v[2] << ' ' << v[3] << ']');
}

inline void Vec4d::convertToArray(double vecArray[4]) const
{
  vecArray[0] = elt[0];
  vecArray[1] = elt[1];
  vecArray[2] = elt[2];
  vecArray[3] = elt[3];
}

inline void Vec4d::addToArray(double vecArray[4]) const
{
  vecArray[0] += elt[0];
  vecArray[1] += elt[1];
  vecArray[2] += elt[2];
  vecArray[3] += elt[3];
}

inline void Vec4d::normalize()
{
  double invMag = 1.0 / len(*this);
  (*this) *= invMag;
}

inline void Vec4d::print() const
{
  printf("[%G %G %G %G]\n", elt[0], elt[1], elt[2], elt[3]);
}

inline void Vec4d::set(double x0, double x1, double x2, double x3)
{
  elt[0] = x0;
  elt[1] = x1;
  elt[2] = x2;
  elt[3] = x3;
}

inline void Vec4d::set(double value)
{
  elt[0] = elt[1] = elt[2] = elt[3] = value;
}

inline bool Vec4d::hasNaN() const
{
  return (std::isnan(elt[0]) || std::isnan(elt[1]) || std::isnan(elt[2]) || std::isnan(elt[3]));
}

#endif

