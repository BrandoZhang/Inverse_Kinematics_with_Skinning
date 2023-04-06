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

  This class implements a 3x3 matrix, with elementary algebra,
  including the computation of eigenvalues and eigenvectors of a *symmetric* 3x3 matrix.
  (using a public domain external routine; see mat3d.cpp and eig3.h)
*/



#ifndef _MINIMAT3D_H_
#define _MINIMAT3D_H_

#include "vec3d.h"

class Mat3d
{
public:
  enum ColumnMajorInput {};
  
  inline Mat3d() {}

  /*
          [ x0  x1  x2 ]
    M  =  [ x3  x4  x5 ]
          [ x6  x7  x8 ]
  */
  inline Mat3d(double x0, double x1, double x2,
               double x3, double x4, double x5,
               double x6, double x7, double x8);
  inline explicit Mat3d(const double mat[9]); // "mat" must be given in row-major order
  inline explicit Mat3d(const Vec3d rows[3]); 
  inline Mat3d(const Vec3d & row0, const Vec3d & row1, const Vec3d & row2);
  inline Mat3d(const double row0[3], const double row1[3], const double row2[3]);
  inline Mat3d(const Vec3d & col0, const Vec3d & col1, const Vec3d & col2, ColumnMajorInput);
  inline Mat3d(const double col0[3], const double col1[3], const double col2[3], ColumnMajorInput);
  inline explicit Mat3d(double diag); // create a diagonal matrix with all entries "diag" (can create zero matrix by passing 0.0)

  inline void set(double x0, double x1, double x2,
		  double x3, double x4, double x5,
		  double x6, double x7, double x8);
  inline void set(double value); // set all elements to value
  inline void set(const double values[9]);

  inline void setDiag(const double value) { elt[0][0] = elt[1][1] = elt[2][2] = value; }
  inline void setDiag(const double values[3]) { elt[0][0] = values[0]; elt[1][1] = values[1]; elt[2][2] = values[2]; }

  inline Mat3d & operator=(const Mat3d & source); 

  inline Mat3d operator+ (const Mat3d & ) const;
  inline Mat3d & operator+= (const Mat3d & );

  inline Mat3d operator- (const Mat3d & ) const;
  inline Mat3d & operator-= (const Mat3d & );

  inline Mat3d & operator*= (double scalar);
  inline Mat3d & operator/= (double scalar);

  inline bool operator== (const Mat3d &) const;
  inline bool operator!= (const Mat3d &) const;

  friend inline Mat3d operator* (double scalar, const Mat3d & mat2);
  friend inline Mat3d operator/ (const Mat3d & mat2, double scalar);

  inline Vec3d transposeMultiply(const Vec3d & v) const; // v' = M^T v

  inline double maxAbsEntry() const;
  inline bool hasNaN() const;
  friend inline std::ostream &operator << (std::ostream &s, const Mat3d &v);
  inline void print() const;

  inline Vec3d & operator[] (int index); // M[i] returns i-th row
  inline const Vec3d & operator[] (int index) const; // M[i] returns i-th row

  inline Vec3d col(int index) const; // get i-th column
  inline Vec3d diag() const { return Vec3d(elt[0][0], elt[1][1], elt[2][2]); }

  // converts the matrix into a C array of length 9 in row-major order
  inline void convertToArray(double array[9]) const;
  inline void addToArray(double array[9]) const;
  inline void addToArray(double vec0[3], double vec1[3], double vec2[3]) const;

  const double * data() const { return &elt[0][0]; }
  double * data() { return &elt[0][0]; }

  // matrix-vector multiply
  inline const Vec3d operator* (const Vec3d & vec) const;

  // matrix-matrix multiply
  inline const Mat3d operator* (const Mat3d & mat2) const;

  const static Mat3d Identity;
  const static Mat3d Zero;
protected:
  Vec3d elt[3]; // the three rows of the matrix
};

// Computes eigenvalues and eigenvectors of a 3x3 matrix M
// Assumes symmetric matrix; contents of matrix "M" are not modified by the routine
// Eigenvalues are sorted in decreasing order (not decreasing absolute-value order)
// Returned eigenvectors are unit length
void eigen_sym(Mat3d & M, Vec3d & eig_val, Vec3d eig_vec[3]=NULL);

// inverse of a 3x3 matrix (suitable when A is given by a pointer to the 9 entries)
// input: A (row-major)
// output: AInv
void inverse3x3(const double * A, double * AInv);

// these two declarations are repeated here, with default arguments, to conform to the C++ standard; this makes it possible to compile the code on Apple's clang compiler
bool eigen_sym_NR(Mat3d & M, Vec3d & eig_val, Vec3d eig_vec[3], int maxIterations=50, double epsilon=0.0);
int SVD(Mat3d & M, Mat3d & U, Vec3d & Sigma, Mat3d & V, double singularValue_eps=1e-8, int modifiedSVD=0);

// return element rotation matrix that rotates around X, Y or Z axis
//      [ 1  0  0]       [ c  0  s]       [ c -s  0] 
// Rx = [ 0  c -s]  Ry = [ 0  1  0]  Rz = [ s  c  0]
//      [ 0  s  c]       [-s  0  c]       [ 0  0  1]
// where c = cos(angle), s = sin(angle)
inline Mat3d getElementRotationMatrix(int axis, double angle); // axis is 0,1,2, angle in radian

inline Mat3d tensorProduct(const Vec3d & vec1, const Vec3d & vec2);

inline Mat3d inv(const Mat3d &); // inverse matrix

inline double det(const Mat3d & mat); // determinant

inline Mat3d trans(const Mat3d & mat); // transpose

// treat the memory pointed by mat as a Mat3d object
inline Mat3d & asMat3d(double * mat) { return *(Mat3d *)mat; }
inline const Mat3d & asMat3d(const double * mat) { return *(const Mat3d *)mat; }
inline Mat3d * asMat3dPtr(double * mat) { return (Mat3d *)mat; }
inline const Mat3d * asMat3dPtr(const double * mat) { return (const Mat3d *)mat; }

// ===== below is the implementation =====

inline Mat3d::Mat3d(double x0_g, double x1_g, double x2_g,
                    double x3_g, double x4_g, double x5_g,
                    double x6_g, double x7_g, double x8_g)
{
  elt[0].set(x0_g,x1_g,x2_g);
  elt[1].set(x3_g,x4_g,x5_g);
  elt[2].set(x6_g,x7_g,x8_g);
}

inline Mat3d::Mat3d(const double mat[9])
{
  elt[0].set(mat[0],mat[1],mat[2]);
  elt[1].set(mat[3],mat[4],mat[5]);
  elt[2].set(mat[6],mat[7],mat[8]);
}

inline Mat3d::Mat3d(const Vec3d rows[3]) 
{
  elt[0] = rows[0];
  elt[1] = rows[1];
  elt[2] = rows[2];
}

inline Mat3d::Mat3d(const Vec3d & row0, const Vec3d & row1, const Vec3d & row2)
{
  elt[0] = row0;
  elt[1] = row1;
  elt[2] = row2;
}

inline Mat3d::Mat3d(const double row0[3], const double row1[3], const double row2[3]) 
{
  elt[0].set(row0);
  elt[1].set(row1);
  elt[2].set(row2);
}

inline Mat3d::Mat3d(const Vec3d & col0, const Vec3d & col1, const Vec3d & col2, ColumnMajorInput)
{
  elt[0].set(col0[0], col1[0], col2[0]);
  elt[1].set(col0[1], col1[1], col2[1]);
  elt[2].set(col0[2], col1[2], col2[2]);
}

inline Mat3d::Mat3d(const double col0[3], const double col1[3], const double col2[3], ColumnMajorInput)
{
  elt[0].set(col0[0], col1[0], col2[0]);
  elt[1].set(col0[1], col1[1], col2[1]);
  elt[2].set(col0[2], col1[2], col2[2]);
}

inline Mat3d::Mat3d(double diag)
{
  elt[0].set(diag,0,0);
  elt[1].set(0,diag,0);
  elt[2].set(0,0,diag);
}

inline void Mat3d::set(double x0, double x1, double x2,
		       double x3, double x4, double x5,
		       double x6, double x7, double x8)
{
  elt[0][0] = x0;
  elt[0][1] = x1;
  elt[0][2] = x2;

  elt[1][0] = x3;
  elt[1][1] = x4;
  elt[1][2] = x5;

  elt[2][0] = x6;
  elt[2][1] = x7;
  elt[2][2] = x8;
}

inline void Mat3d::set(const double values[9])
{
  memcpy(data(), values, sizeof(double) * 9);
}

inline void Mat3d::set(double value)
{
  elt[0][0] = elt[0][1] = elt[0][2] = 
  elt[1][0] = elt[1][1] = elt[1][2] =
  elt[2][0] = elt[2][1] = elt[2][2] = value;
}

inline Mat3d & Mat3d::operator=(const Mat3d & source)
{
  elt[0] = source.elt[0];
  elt[1] = source.elt[1];
  elt[2] = source.elt[2];

  return *this;
}

inline Mat3d Mat3d::operator+ (const Mat3d & mat2) const
{
  Mat3d sum = *this;
  sum.elt[0] += mat2.elt[0];
  sum.elt[1] += mat2.elt[1];
  sum.elt[2] += mat2.elt[2];

  return sum;
}

inline Mat3d & Mat3d::operator+= (const Mat3d & mat2)
{
  elt[0] += mat2.elt[0];
  elt[1] += mat2.elt[1];
  elt[2] += mat2.elt[2];
  return *this;
}

inline Mat3d Mat3d::operator- (const Mat3d & mat2) const
{
  Mat3d sum = *this;
  sum.elt[0] -= mat2.elt[0];
  sum.elt[1] -= mat2.elt[1];
  sum.elt[2] -= mat2.elt[2];

  return sum;
}

inline Mat3d & Mat3d::operator-= (const Mat3d & mat2)
{
  elt[0] -= mat2.elt[0];
  elt[1] -= mat2.elt[1];
  elt[2] -= mat2.elt[2];

  return *this;
}

inline Vec3d & Mat3d::operator[] (int index)
{
  return elt[index];
}

inline const Vec3d & Mat3d::operator[] (int index) const
{
  return elt[index];
}

inline Vec3d Mat3d::col(int index) const
{
  return Vec3d(elt[0][index], elt[1][index], elt[2][index]);
}

inline Mat3d & Mat3d::operator*= (double scalar)
{
  elt[0] *= scalar;
  elt[1] *= scalar;
  elt[2] *= scalar;

  return *this;
}

inline bool Mat3d::operator== (const Mat3d & mat2) const
{
  return elt[0] == mat2.elt[0] && elt[1] == mat2.elt[1] && elt[2] == mat2.elt[2]; 
}

inline bool Mat3d::operator!= (const Mat3d & mat2) const
{
  return elt[0] != mat2.elt[0] || elt[1] != mat2.elt[1] || elt[2] != mat2.elt[2];
}

inline Mat3d operator* (double scalar, const Mat3d & mat2)
{
  Mat3d result = mat2;
  result.elt[0] *= scalar;
  result.elt[1] *= scalar;
  result.elt[2] *= scalar;

  return result;
}

inline Mat3d operator/ (const Mat3d & mat2, double scalar)
{
  Mat3d result = mat2;
  result.elt[0] /= scalar;
  result.elt[1] /= scalar;
  result.elt[2] /= scalar;

  return result;
}
 
inline Mat3d tensorProduct(const Vec3d & vecA, const Vec3d & vecB)
{
  Mat3d result(vecA[0]*vecB[0],vecA[0]*vecB[1],vecA[0]*vecB[2],
	       vecA[1]*vecB[0],vecA[1]*vecB[1],vecA[1]*vecB[2],
	       vecA[2]*vecB[0],vecA[2]*vecB[1],vecA[2]*vecB[2]);

  return result;
}

inline Mat3d & Mat3d::operator/= (double scalar)
{
  elt[0] /= scalar;
  elt[1] /= scalar;
  elt[2] /= scalar;

  return *this;
}

inline const Vec3d Mat3d::operator* (const Vec3d & vec) const
{
  return(Vec3d(
    dot(elt[0],vec),
    dot(elt[1],vec),
    dot(elt[2],vec)));
}

inline const Mat3d Mat3d::operator* (const Mat3d & mat2) const
{
  return(Mat3d(
    dot(elt[0],Vec3d(mat2.elt[0][0],mat2.elt[1][0],mat2.elt[2][0])),
    dot(elt[0],Vec3d(mat2.elt[0][1],mat2.elt[1][1],mat2.elt[2][1])),
    dot(elt[0],Vec3d(mat2.elt[0][2],mat2.elt[1][2],mat2.elt[2][2])),

    dot(elt[1],Vec3d(mat2.elt[0][0],mat2.elt[1][0],mat2.elt[2][0])),
    dot(elt[1],Vec3d(mat2.elt[0][1],mat2.elt[1][1],mat2.elt[2][1])),
    dot(elt[1],Vec3d(mat2.elt[0][2],mat2.elt[1][2],mat2.elt[2][2])),

    dot(elt[2],Vec3d(mat2.elt[0][0],mat2.elt[1][0],mat2.elt[2][0])),
    dot(elt[2],Vec3d(mat2.elt[0][1],mat2.elt[1][1],mat2.elt[2][1])),
    dot(elt[2],Vec3d(mat2.elt[0][2],mat2.elt[1][2],mat2.elt[2][2])) ));
}

inline Mat3d operator- (const Mat3d & mat)
{
  return (-1) * mat;
}

inline Mat3d inv(const Mat3d & mat)
{
  Mat3d AInvMat;
  inverse3x3(mat.data(), AInvMat.data());
  return AInvMat;
}

// inverse of a 3x3 matrix
// row-major format
inline void inverse3x3(const double * A, double * AInv)
{
  // converted to C from Mathematica output   
  AInv[0] = -A[5] * A[7] + A[4] * A[8];
  AInv[1] = A[2] * A[7] - A[1] * A[8];
  AInv[2] = -A[2] * A[4] + A[1] * A[5];
  AInv[3] = A[5] * A[6] - A[3] * A[8];
  AInv[4] = -A[2] * A[6] + A[0] * A[8];
  AInv[5] = A[2] * A[3] - A[0] * A[5];
  AInv[6] = -A[4] * A[6] + A[3] * A[7];
  AInv[7] = A[1] * A[6] - A[0] * A[7];
  AInv[8] = -A[1] * A[3] + A[0] * A[4];

  double invDet = 1.0 / (A[0] * AInv[0] + A[1] * AInv[3] + A[2] * AInv[6]);

  for(int i=0; i<9; i++)
    AInv[i] *= invDet;
}

inline double det(const Mat3d & mat) 
{
  return
   (-mat[0][2] * mat[1][1] * mat[2][0] + 
     mat[0][1] * mat[1][2] * mat[2][0] + 
     mat[0][2] * mat[1][0] * mat[2][1] - 
     mat[0][0] * mat[1][2] * mat[2][1] - 
     mat[0][1] * mat[1][0] * mat[2][2] + 
     mat[0][0] * mat[1][1] * mat[2][2] );
}

inline Mat3d trans(const Mat3d & mat)
{
  return
       Mat3d( mat[0][0], mat[1][0], mat[2][0],
              mat[0][1], mat[1][1], mat[2][1],
              mat[0][2], mat[1][2], mat[2][2] );
}

// compute [  0  -v2  v1 ]
//         [  v2  0  -v0 ]
//         [ -v1  v0  0  ]
inline Mat3d skewSymmetricMatrix(const Vec3d & vec)
{
  return Mat3d(0, -vec[2], vec[1], 
               vec[2], 0, -vec[0],
               -vec[1], vec[0], 0);
}

inline void Mat3d::convertToArray(double array[9]) const // in row-major order
{
  memcpy(array, data(), sizeof(double) * 9);
}

inline void Mat3d::addToArray(double array[9]) const // in row-major order
{
  array[0] += elt[0][0]; array[1] += elt[0][1]; array[2] += elt[0][2];
  array[3] += elt[1][0]; array[4] += elt[1][1]; array[5] += elt[1][2];
  array[6] += elt[2][0]; array[7] += elt[2][1]; array[8] += elt[2][2];
}

inline void Mat3d::addToArray(double vec0[3], double vec1[3], double vec2[3]) const
{
  elt[0].addToArray(vec0);
  elt[1].addToArray(vec1);
  elt[2].addToArray(vec2);
}

inline std::ostream &operator << (std::ostream &s, const Mat3d &v)
{
  return s << '[' << v[0] << ' ' << v[1] << ' ' << v[2] << ']';
}

inline void Mat3d::print() const
{
  double a00 = elt[0][0]; double a01 = elt[0][1]; double a02 = elt[0][2];
  double a10 = elt[1][0]; double a11 = elt[1][1]; double a12 = elt[1][2];
  double a20 = elt[2][0]; double a21 = elt[2][1]; double a22 = elt[2][2];
  
  printf("[%G %G %G;\n", a00, a01, a02);
  printf(" %G %G %G;\n", a10, a11, a12);
  printf(" %G %G %G]\n", a20, a21, a22);
}

inline double Mat3d::maxAbsEntry() const
{
  double maxAbsEntry = 0.0;
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
    {
       if (fabs(elt[i][j]) > maxAbsEntry)
         maxAbsEntry = fabs(elt[i][j]);
    }
  return maxAbsEntry;
}

inline bool Mat3d::hasNaN() const
{
  return elt[0].hasNaN() || elt[1].hasNaN() || elt[2].hasNaN();
}

//      [ 1  0  0]       [ c  0  s]       [ c -s  0] 
// Rx = [ 0  c -s]  Ry = [ 0  1  0]  Rz = [ s  c  0]
//      [ 0  s  c]       [-s  0  c]       [ 0  0  1]
inline Mat3d getElementRotationMatrix(int axis, double angle) // axis is 0,1,2, angle in radian
{
  double c = cos(angle), s = sin(angle);
  switch(axis)
  {
    case 0:
      return Mat3d(1, 0, 0, 0, c, -s, 0, s, c);
    case 1:
      return Mat3d(c, 0, s, 0, 1, 0, -s, 0, c);
    case 2:
      return Mat3d(c, -s, 0, s, c, 0, 0, 0, 1);
  }
  // invalid case
  return Mat3d(0.0);
}

#endif

