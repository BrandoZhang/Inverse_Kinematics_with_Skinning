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

  This class implements a 4x4 matrix, with elementary algebra
*/



#ifndef _MINIMAT4D_H_
#define _MINIMAT4D_H_

#include "vec4d.h"

class Mat4d
{
public:
  
  inline Mat4d() {}

  /*
          [  x0  x1  x2  x3 ]
    M  =  [  x4  x5  x6  x7 ]
          [  x8  x9 x10 x11 ]
          [ x12 x13 x14 x15 ]
  */
  inline Mat4d(double  x0, double  x1, double  x2, double  x3,
               double  x4, double  x5, double  x6, double  x7,
               double  x8, double  x9, double x10, double x11,
               double x12, double x13, double x14, double x15);
  inline Mat4d(const double mat[16]); // "mat" must be given in row-major order
  inline Mat4d(const Vec4d rows[4]); 
  inline Mat4d(const Vec4d & row0, const Vec4d & row1, const Vec4d & row2, const Vec4d & row3);
  inline Mat4d(double diag); // create a diagonal matrix with all entries "diag" (can create zero matrix by passing 0.0)

  inline void set(double  x0, double  x1, double  x2, double  x3,
                 double  x4, double  x5, double  x6, double  x7,
                 double  x8, double  x9, double x10, double x11,
                 double x12, double x13, double x14, double x15);
  inline void set(double value); // set all elements to value

  inline Mat4d & operator=(const Mat4d & source); 

  inline Mat4d operator+ (const Mat4d & ) const;
  inline Mat4d & operator+= (const Mat4d & );

  inline Mat4d operator- (const Mat4d & ) const;
  inline Mat4d & operator-= (const Mat4d & );

  inline Mat4d & operator*= (double scalar);
  inline Mat4d & operator/= (double scalar);

  inline bool operator== (const Mat4d & mat2) const;
  inline bool operator!= (const Mat4d & mat2) const { return !(*this == mat2); }

  friend inline Mat4d operator* (double scalar, const Mat4d & mat2);
  friend inline Mat4d operator/ (const Mat4d & mat2, double scalar);

  inline Mat4d & multiplyDiagRight(Vec4d & v); // M = M * diag(v)
  inline Mat4d & multiplyDiagLeft(Vec4d & v); // M = diag(v) * M

  friend inline Mat4d tensorProduct(const Vec4d & vec1, const Vec4d & vec2);

  friend inline Mat4d inv(const Mat4d &); // inverse matrix
  friend inline double det(const Mat4d & mat); // determinant
  friend inline Mat4d trans(const Mat4d & mat); // transpose

  inline double maxAbsEntry() const;
  inline bool hasNaN() const;
  friend inline std::ostream &operator << (std::ostream &s, const Mat4d &v);
  inline void print() const;

  inline Vec4d & operator[] (int index) { return elt[index]; } // M[i] returns i-th row
  inline const Vec4d & operator[] (int index) const { return elt[index]; } // M[i] returns i-th row

  inline Vec4d col(int index) const { return Vec4d(elt[0][index], elt[1][index], elt[2][index], elt[3][index]); } // return i-th col

  // converts the matrix into a C array of length 16 in row-major order
  inline void convertToArray(double array[16]) const;
  inline void addToArray(double array[16]) const;

  const double * data() const { return &elt[0][0]; }
  double * data() { return &elt[0][0]; }

  // matrix-vector multiply
  inline const Vec4d operator* (const Vec4d & vec) const;

  // matrix-matrix multiply
  inline const Mat4d operator* (const Mat4d & mat2) const;

  const static Mat4d Identity;
  const static Mat4d Zero;
protected:
  Vec4d elt[4]; // the four rows of the matrix
};

// inverse of a 3x3 matrix (suitable when A is given by a pointer to the 9 entries)
// input: A (row-major)
// output: AInv (row-major)
inline void inverse4x4(const double A[16], double AInv[16]);

// treat the memory pointed by mat as a Mat4d object
inline Mat4d & asMat4d(double * mat) { return *(Mat4d *)mat; }
inline const Mat4d & asMat4d(const double * mat) { return *(const Mat4d *)mat; }
inline Mat4d * asMat4dPtr(double * mat) { return (Mat4d *)mat; }
inline const Mat4d * asMat4dPtr(const double * mat) { return (const Mat4d *)mat; }

// ===== below is the implementation =====

inline Mat4d::Mat4d(double  x0, double  x1, double  x2, double  x3,
               double  x4, double  x5, double  x6, double  x7,
               double  x8, double  x9, double x10, double x11,
               double x12, double x13, double x14, double x15)
{
  elt[0] = Vec4d(x0,x1,x2,x3);
  elt[1] = Vec4d(x4,x5,x6,x7);
  elt[2] = Vec4d(x8,x9,x10,x11);
  elt[3] = Vec4d(x12,x13,x14,x15);
}

inline Mat4d::Mat4d(const double mat[16])
{
  elt[0] = Vec4d(mat[0],mat[1],mat[2],mat[3]);
  elt[1] = Vec4d(mat[4],mat[5],mat[6],mat[7]);
  elt[2] = Vec4d(mat[8],mat[9],mat[10],mat[11]);
  elt[3] = Vec4d(mat[12],mat[13],mat[14],mat[15]);
}

inline Mat4d::Mat4d(const Vec4d rows[4]) 
{
  elt[0] = rows[0];
  elt[1] = rows[1];
  elt[2] = rows[2];
  elt[3] = rows[3];
}

inline Mat4d::Mat4d(const Vec4d & row0, const Vec4d & row1, const Vec4d & row2, const Vec4d & row3)
{
  elt[0] = row0;
  elt[1] = row1;
  elt[2] = row2;
  elt[3] = row3;
}

inline Mat4d::Mat4d(double diag)
{
  elt[0] = Vec4d(diag,0,0,0);
  elt[1] = Vec4d(0,diag,0,0);
  elt[2] = Vec4d(0,0,diag,0);
  elt[3] = Vec4d(0,0,0,diag);
}

inline void Mat4d::set(double  x0, double  x1, double  x2, double  x3,
               double  x4, double  x5, double  x6, double  x7,
               double  x8, double  x9, double x10, double x11,
               double x12, double x13, double x14, double x15)
{
  elt[0] = Vec4d(x0,x1,x2,x3);
  elt[1] = Vec4d(x4,x5,x6,x7);
  elt[2] = Vec4d(x8,x9,x10,x11);
  elt[3] = Vec4d(x12,x13,x14,x15);
}

inline void Mat4d::set(double value)
{
  elt[0].set(value);
  elt[1].set(value);
  elt[2].set(value);
  elt[3].set(value);
}

inline Mat4d & Mat4d::operator=(const Mat4d & source)
{
  if (&source != this)
    memcpy(elt, source.elt, sizeof(double) * 16);
  return *this;
}

inline Mat4d Mat4d::operator+ (const Mat4d & mat2) const
{
  Mat4d sum = *this;
  sum.elt[0] += mat2.elt[0];
  sum.elt[1] += mat2.elt[1];
  sum.elt[2] += mat2.elt[2];
  sum.elt[3] += mat2.elt[3];
  return sum;
}

inline Mat4d & Mat4d::operator+= (const Mat4d & mat2)
{
  elt[0] += mat2.elt[0];
  elt[1] += mat2.elt[1];
  elt[2] += mat2.elt[2];
  elt[3] += mat2.elt[3];
  return *this;
}

inline Mat4d Mat4d::operator- (const Mat4d & mat2) const
{
  Mat4d sum = *this;
  sum.elt[0] -= mat2.elt[0];
  sum.elt[1] -= mat2.elt[1];
  sum.elt[2] -= mat2.elt[2];
  sum.elt[3] -= mat2.elt[3];
  return sum;
}

inline Mat4d & Mat4d::operator-= (const Mat4d & mat2)
{
  elt[0] -= mat2.elt[0];
  elt[1] -= mat2.elt[1];
  elt[2] -= mat2.elt[2];
  elt[3] -= mat2.elt[3];
  return *this;
}

inline Mat4d & Mat4d::operator*= (double scalar)
{
  elt[0] *= scalar;
  elt[1] *= scalar;
  elt[2] *= scalar;
  elt[3] *= scalar;
  return *this;
}

inline bool Mat4d::operator== (const Mat4d & mat2) const
{
  return elt[0] == mat2.elt[0] && elt[1] == mat2.elt[1] && elt[2] == mat2.elt[2] && elt[3] == mat2.elt[3]; 
}

inline Mat4d & Mat4d::multiplyDiagRight(Vec4d & v) // M = M * diag(v)
{
  elt[0][0] *= v[0];
  elt[1][0] *= v[0];
  elt[2][0] *= v[0];
  elt[3][0] *= v[0];

  elt[0][1] *= v[1];
  elt[1][1] *= v[1];
  elt[2][1] *= v[1];
  elt[3][1] *= v[1];

  elt[0][2] *= v[2];
  elt[1][2] *= v[2];
  elt[2][2] *= v[2];
  elt[3][2] *= v[2];

  elt[0][3] *= v[3];
  elt[1][3] *= v[3];
  elt[2][3] *= v[3];
  elt[3][3] *= v[3];
  return *this;
}

inline Mat4d & Mat4d::multiplyDiagLeft(Vec4d & v) // M = diag(v) * M
{
  elt[0] *= v[0];
  elt[1] *= v[1];
  elt[2] *= v[2];
  elt[3] *= v[3];

  return *this;
}

inline Mat4d operator* (double scalar, const Mat4d & mat2)
{
  Mat4d result = mat2;
  result.elt[0] *= scalar;
  result.elt[1] *= scalar;
  result.elt[2] *= scalar;
  result.elt[3] *= scalar;

  return result;
}

inline Mat4d operator/ (const Mat4d & mat2, double scalar)
{
  Mat4d result = mat2;
  result.elt[0] /= scalar;
  result.elt[1] /= scalar;
  result.elt[2] /= scalar;
  result.elt[3] /= scalar;

  return result;
}
 
inline Mat4d tensorProduct(const Vec4d & vecA, const Vec4d & vecB)
{
  Mat4d result(vecA[0]*vecB[0],vecA[0]*vecB[1],vecA[0]*vecB[2],vecA[0]*vecB[3],
	             vecA[1]*vecB[0],vecA[1]*vecB[1],vecA[1]*vecB[2],vecA[1]*vecB[3],
	             vecA[2]*vecB[0],vecA[2]*vecB[1],vecA[2]*vecB[2],vecA[2]*vecB[3],
               vecA[3]*vecB[0],vecA[3]*vecB[1],vecA[3]*vecB[2],vecA[3]*vecB[3]);

  return result;
}

inline Mat4d & Mat4d::operator/= (double scalar)
{
  elt[0] /= scalar;
  elt[1] /= scalar;
  elt[2] /= scalar;
  elt[3] /= scalar;

  return *this;
}

inline const Vec4d Mat4d::operator* (const Vec4d & vec) const
{
  return(Vec4d(
    dot(elt[0],vec),
    dot(elt[1],vec),
    dot(elt[2],vec),
    dot(elt[3],vec)));
}

inline const Mat4d Mat4d::operator* (const Mat4d & mat2) const
{
  return(Mat4d(
    dot(elt[0],Vec4d(mat2.elt[0][0],mat2.elt[1][0],mat2.elt[2][0], mat2.elt[3][0])),
    dot(elt[0],Vec4d(mat2.elt[0][1],mat2.elt[1][1],mat2.elt[2][1], mat2.elt[3][1])),
    dot(elt[0],Vec4d(mat2.elt[0][2],mat2.elt[1][2],mat2.elt[2][2], mat2.elt[3][1])),
    dot(elt[0],Vec4d(mat2.elt[0][3],mat2.elt[1][3],mat2.elt[2][3], mat2.elt[3][3])),

    dot(elt[1],Vec4d(mat2.elt[0][0],mat2.elt[1][0],mat2.elt[2][0], mat2.elt[3][0])),
    dot(elt[1],Vec4d(mat2.elt[0][1],mat2.elt[1][1],mat2.elt[2][1], mat2.elt[3][1])),
    dot(elt[1],Vec4d(mat2.elt[0][2],mat2.elt[1][2],mat2.elt[2][2], mat2.elt[3][1])),
    dot(elt[1],Vec4d(mat2.elt[0][3],mat2.elt[1][3],mat2.elt[2][3], mat2.elt[3][3])),

    dot(elt[2],Vec4d(mat2.elt[0][0],mat2.elt[1][0],mat2.elt[2][0], mat2.elt[3][0])),
    dot(elt[2],Vec4d(mat2.elt[0][1],mat2.elt[1][1],mat2.elt[2][1], mat2.elt[3][1])),
    dot(elt[2],Vec4d(mat2.elt[0][2],mat2.elt[1][2],mat2.elt[2][2], mat2.elt[3][1])),
    dot(elt[2],Vec4d(mat2.elt[0][3],mat2.elt[1][3],mat2.elt[2][3], mat2.elt[3][3])),

    dot(elt[3],Vec4d(mat2.elt[0][0],mat2.elt[1][0],mat2.elt[2][0], mat2.elt[3][0])),
    dot(elt[3],Vec4d(mat2.elt[0][1],mat2.elt[1][1],mat2.elt[2][1], mat2.elt[3][1])),
    dot(elt[3],Vec4d(mat2.elt[0][2],mat2.elt[1][2],mat2.elt[2][2], mat2.elt[3][1])),
    dot(elt[3],Vec4d(mat2.elt[0][3],mat2.elt[1][3],mat2.elt[2][3], mat2.elt[3][3]))));
}

inline Mat4d inv(const Mat4d & mat)
{
  Mat4d mat2;
  inverse4x4(mat.data(), mat2.data());

  return mat2;
}

inline void inverse4x4(const double A[16], double AInv[16])
{
  // converted to C from Mathematica output   
  AInv[0] = -A[11] * A[14] * A[5] + A[10] * A[15] * A[5] + A[11] * A[13] * A[6] - A[10] * A[13] * A[7] - A[15] * A[6] * A[9] + A[14] * A[7] * A[9];
  AInv[1] = A[1] * A[11] * A[14] - A[1] * A[10] * A[15] - A[11] * A[13] * A[2] + A[10] * A[13] * A[3] + A[15] * A[2] * A[9] - A[14] * A[3] * A[9];
  AInv[2] = -A[15] * A[2] * A[5] + A[14] * A[3] * A[5] + A[1] * A[15] * A[6] - A[13] * A[3] * A[6] - A[1] * A[14] * A[7] + A[13] * A[2] * A[7];
  AInv[3] = A[11] * A[2] * A[5] - A[10] * A[3] * A[5] - A[1] * A[11] * A[6] + A[1] * A[10] * A[7] + A[3] * A[6] * A[9] - A[2] * A[7] * A[9];
  AInv[4] = A[11] * A[14] * A[4] - A[10] * A[15] * A[4] - A[11] * A[12] * A[6] + A[10] * A[12] * A[7] + A[15] * A[6] * A[8] - A[14] * A[7] * A[8];
  AInv[5] = -A[0] * A[11] * A[14] + A[0] * A[10] * A[15] + A[11] * A[12] * A[2] - A[10] * A[12] * A[3] - A[15] * A[2] * A[8] + A[14] * A[3] * A[8];
  AInv[6] = A[15] * A[2] * A[4] - A[14] * A[3] * A[4] - A[0] * A[15] * A[6] + A[12] * A[3] * A[6] + A[0] * A[14] * A[7] - A[12] * A[2] * A[7];
  AInv[7] = -A[11] * A[2] * A[4] + A[10] * A[3] * A[4] + A[0] * A[11] * A[6] - A[0] * A[10] * A[7] - A[3] * A[6] * A[8] + A[2] * A[7] * A[8];
  AInv[8] = -A[11] * A[13] * A[4] + A[11] * A[12] * A[5] - A[15] * A[5] * A[8] + A[13] * A[7] * A[8] + A[15] * A[4] * A[9] - A[12] * A[7] * A[9];
  AInv[9] = -A[1] * A[11] * A[12] + A[0] * A[11] * A[13] + A[1] * A[15] * A[8] - A[13] * A[3] * A[8] - A[0] * A[15] * A[9] + A[12] * A[3] * A[9];
  AInv[10] = -A[1] * A[15] * A[4] + A[13] * A[3] * A[4] + A[0] * A[15] * A[5] - A[12] * A[3] * A[5] + A[1] * A[12] * A[7] - A[0] * A[13] * A[7];
  AInv[11] = A[1] * A[11] * A[4] - A[0] * A[11] * A[5] + A[3] * A[5] * A[8] - A[1] * A[7] * A[8] - A[3] * A[4] * A[9] + A[0] * A[7] * A[9];
  AInv[12] = A[10] * A[13] * A[4] - A[10] * A[12] * A[5] + A[14] * A[5] * A[8] - A[13] * A[6] * A[8] - A[14] * A[4] * A[9] + A[12] * A[6] * A[9];
  AInv[13] = A[1] * A[10] * A[12] - A[0] * A[10] * A[13] - A[1] * A[14] * A[8] + A[13] * A[2] * A[8] + A[0] * A[14] * A[9] - A[12] * A[2] * A[9];
  AInv[14] = A[1] * A[14] * A[4] - A[13] * A[2] * A[4] - A[0] * A[14] * A[5] + A[12] * A[2] * A[5] - A[1] * A[12] * A[6] + A[0] * A[13] * A[6];
  AInv[15] = -A[1] * A[10] * A[4] + A[0] * A[10] * A[5] - A[2] * A[5] * A[8] + A[1] * A[6] * A[8] + A[2] * A[4] *A[9] - A[0] * A[6] * A[9];

  double invDet = 1.0 / (A[0] * AInv[0] + A[1] * AInv[4] + A[2] * AInv[8] + A[3] * AInv[12]);

  for(int i=0; i<16; i++)
    AInv[i] *= invDet;
}

inline double det(const Mat4d & mat) 
{
  const double * A = mat.data();
  double AInv0 = -A[11] * A[14] * A[5] + A[10] * A[15] * A[5] + A[11] * A[13] * A[6] - A[10] * A[13] * A[7] - A[15] * A[6] * A[9] + A[14] * A[7] * A[9];
  double AInv4 = A[11] * A[14] * A[4] - A[10] * A[15] * A[4] - A[11] * A[12] * A[6] + A[10] * A[12] * A[7] + A[15] * A[6] * A[8] - A[14] * A[7] * A[8];
  double AInv8 = -A[11] * A[13] * A[4] + A[11] * A[12] * A[5] - A[15] * A[5] * A[8] + A[13] * A[7] * A[8] + A[15] * A[4] * A[9] - A[12] * A[7] * A[9];
  double AInv12 = A[10] * A[13] * A[4] - A[10] * A[12] * A[5] + A[14] * A[5] * A[8] - A[13] * A[6] * A[8] - A[14] * A[4] * A[9] + A[12] * A[6] * A[9];
  return (A[0] * AInv0 + A[1] * AInv4 + A[2] * AInv8 + A[3] * AInv12);
}

inline Mat4d trans(const Mat4d & mat)
{
  return
       Mat4d( mat[0][0], mat[1][0], mat[2][0], mat[3][0],
              mat[0][1], mat[1][1], mat[2][1], mat[3][1],
              mat[0][2], mat[1][2], mat[2][2], mat[3][2],
              mat[0][3], mat[1][3], mat[2][3], mat[3][3]);
}

inline void Mat4d::convertToArray(double array[16]) const // in row-major order
{
  memcpy(array, data(), sizeof(double) * 16);
}

inline void Mat4d::addToArray(double array[16]) const // in row-major order
{
  for(int i = 0; i < 16; i++)
    array[i] += data()[i];
}

inline std::ostream &operator << (std::ostream &s, const Mat4d &v)
{
  return s << '[' << v[0] << ' ' << v[1] << ' ' << v[2] << ' ' << v[3] << ']';
}

inline void Mat4d::print() const
{
  printf("[%G %G %G %G;\n", elt[0][0], elt[0][1], elt[0][2], elt[0][3]);
  printf(" %G %G %G %G;\n", elt[1][0], elt[1][1], elt[1][2], elt[1][3]);
  printf(" %G %G %G %G;\n", elt[2][0], elt[2][1], elt[2][2], elt[2][3]);
  printf(" %G %G %G %G]\n", elt[3][0], elt[3][1], elt[3][2], elt[3][3]);
}

inline double Mat4d::maxAbsEntry() const
{
  double maxAbsEntry = 0.0;
  for(int i = 0; i < 4; i++)
    for(int j = 0; j < 4; j++)
    {
       if (fabs(elt[i][j]) > maxAbsEntry)
         maxAbsEntry = fabs(elt[i][j]);
    }
  return maxAbsEntry;
}

inline bool Mat4d::hasNaN() const
{
  return elt[0].hasNaN() || elt[1].hasNaN() || elt[2].hasNaN() || elt[3].hasNaN();
}

#endif

