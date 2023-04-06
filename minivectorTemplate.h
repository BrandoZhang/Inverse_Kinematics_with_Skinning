#ifndef MINIVECTORTEMPLATE_H
#define MINIVECTORTEMPLATE_H

#include <stdio.h>
#include <math.h>

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

// A templated wrapper (Vec3: 3-vectors; Mat3: 3x3 matrices).
// Used for custom floating-point values like traced values (adouble in Adol-C),
// or exact/high-precision arithmetic values (exact kernels in CGAL).
template <typename real>
class Vec3
{
public:
  inline Vec3() {}
  Vec3(const real & x, const real & y, const real & z) { v[0]=x; v[1]=y; v[2]=z; }
  explicit Vec3(const real & value) { v[0] = v[1] = v[2] = value; }
  template <typename real2>
  explicit Vec3(const real2 w[3]) { v[0] = w[0]; v[1] = w[1]; v[2] = w[2]; }
  Vec3(const Vec3 & w) { v[0] = w[0]; v[1] = w[1]; v[2] = w[2]; }

  template <typename real2>
  void set(const real2 & x, const real2 & y, const real2 & z) { v[0]=x; v[1]=y; v[2]=z; }
  template <typename real2>
  void set(const real & s) { v[0] = v[1] = v[2] = s; }
  template <typename real2>
  void set(const real2 w[3]) { v[0] = w[0]; v[1] = w[1]; v[2] = w[2]; }

  Vec3 & operator=(const Vec3 & w) { v[0] = w[0]; v[1] = w[1]; v[2] = w[2]; return *this; }
  bool operator==(const Vec3 & w) const { return (v[0] == w[0]) && (v[1] == w[1]) && (v[2] == w[2]); }
  bool operator!=(const Vec3 & w) const { return !(*this == w); }

  Vec3 operator+ (const Vec3 & w) const { return Vec3(v[0]+w[0], v[1]+w[1], v[2]+w[2]); }
  Vec3 & operator+= (const Vec3 & w) { v[0] += w[0]; v[1] += w[1]; v[2] += w[2]; return *this; }

  Vec3 operator- (const Vec3 & w) const { return Vec3(v[0]-w[0], v[1]-w[1], v[2]-w[2]); }
  Vec3 & operator-= (const Vec3 & w) { v[0] -= w[0]; v[1] -= w[1]; v[2] -= w[2]; return *this; }

  Vec3 operator* (real s) const { return Vec3(v[0]*s, v[1]*s, v[2]*s); }
  Vec3 & operator*= (real s) { v[0] *= s; v[1] *= s; v[2] *= s; return *this; }
  inline friend Vec3 operator* (real s, const Vec3 & v) { return Vec3(v[0]*s, v[1]*s, v[2]*s); }

  Vec3 operator/ (real s) const { return Vec3(v[0]/s, v[1]/s, v[2]/s); }
  Vec3 & operator/= (real s) { v[0] /= s; v[1] /= s; v[2] /= s; return *this; }

  inline real & operator[] (int index) { return v[index]; }
  inline const real & operator[] (int index) const { return v[index]; }

  const real * data() const { return &v[0]; }
  real * data() { return &v[0]; }

  // Copies the vector into an array of length 3.
  template <typename real2>
  inline void convertToArray(real2 w[3]) const { w[0] = v[0]; w[1] = v[1]; w[2] = v[2]; }
  template <typename real2>
  inline void addToArray(real2 w[3]) const { w[0] += v[0]; w[1] += v[1]; w[2] += v[2]; }

  inline friend Vec3 operator- (const Vec3 & v) { return real(-1) * v; }
  inline friend real dot(const Vec3 & u, const Vec3 & w) { return u[0] * w[0] + u[1] * w[1] + u[2] * w[2]; }
  inline friend Vec3 cross(const Vec3 & u, const Vec3 & w) { return Vec3(u[1] * w[2] - w[1] * u[2], -u[0] * w[2] + w[0] * u[2], u[0] * w[1] - w[0] * u[1]); }

  inline friend real len(const Vec3 & vec) { return sqrt(dot(vec,vec)); }
  inline friend real len2(const Vec3 & vec) { return dot(vec,vec); }

  inline friend Vec3 norm(const Vec3 & vec) { return vec / sqrt(len2(vec)); }
  // normalize itself
  inline void normalize() { real invMag = real(1) / sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); *this *= invMag; }

protected:
  real v[3];
};

template<typename real>
class Mat3
{
public:
  enum ColumnMajorInput {};

  inline Mat3() {}

  /*
          [ x0  x1  x2 ]
    M  =  [ x3  x4  x5 ]
          [ x6  x7  x8 ]
  */
  Mat3(const real & x0, const real & x1, const real & x2,
       const real & x3, const real & x4, const real & x5,
       const real & x6, const real & x7, const real & x8) { v[0].set(x0,x1,x2); v[1].set(x3,x4,x5); v[2].set(x6,x7,x8); }
  // "mat" must be given in row-major order
  template <typename real2>
  explicit Mat3(const real2 mat[9]) { v[0].set(mat); v[1].set(mat+3); v[2].set(mat+6); }
  explicit Mat3(const Vec3<real> rows[3]) { v[0] = rows[0]; v[1] = rows[1]; v[2] = rows[2]; }
  Mat3(const Vec3<real> & row0, const Vec3<real> & row1, const Vec3<real> & row2) { v[0] = row0; v[1] = row1; v[2] = row2; }
  template <typename real2>
  Mat3(const real2 row0[3], const real2 row1[3], const real2 row2[3]) { v[0].set(row0); v[1].set(row1); v[2].set(row2); }
  Mat3(const Vec3<real> & col0, const Vec3<real> & col1, const Vec3<real> & col2, ColumnMajorInput) { v[0].set(col0[0], col1[0], col2[0]); v[1].set(col0[1], col1[1], col2[1]); v[2].set(col0[2], col1[2], col2[2]); }
  template <typename real2>
  Mat3(const real2 col0[3], const real2 col1[3], const real2 col2[3], ColumnMajorInput) { v[0].set(col0[0], col1[0], col2[0]); v[1].set(col0[1], col1[1], col2[1]); v[2].set(col0[2], col1[2], col2[2]); }
  // create a diagonal matrix with all entries "diag" (can create zero matrix by passing 0.0)
  explicit Mat3(real diag) { v[0].set(diag,real(0),real(0)); v[1].set(real(0),diag,real(0)); v[2].set(real(0),real(0),diag); }

  void set(real x0, real x1, real x2, real x3, real x4, real x5, real x6, real x7, real x8) { v[0].set(x0,x1,x2); v[1].set(x3,x4,x5); v[2].set(x6,x7,x8); }
  // set all elements to value
  void set(real value) { v[0].set(value); v[1].set(value); v[2].set(value); }
  void set(const real mat[9]) { v[0].set(mat); v[1].set(mat+3); v[2].set(mat+6); }

  void setDiag(const real value) { v[0][0] = v[1][1] = v[2][2] = value; }
  void setDiag(const real values[3]) { v[0][0] = values[0]; v[1][1] = values[1]; v[2][2] = values[2]; }

  Mat3 & operator=(const Mat3 & w) { v[0] = w[0]; v[1] = w[1]; v[2] = w[2]; return *this; }
  bool operator==(const Mat3 & w) const { return (v[0] == w[0]) && (v[1] == w[1]) && (v[2] == w[2]); }
  bool operator!=(const Mat3 & w) const { return !(*this == w); }

  Mat3 operator+ (const Mat3 & w) const { return Mat3(v[0]+w[0], v[1]+w[1], v[2]+w[2]); }
  Mat3 & operator+= (const Mat3 & w) { v[0] += w[0]; v[1] += w[1]; v[2] += w[2]; return *this; }

  Mat3 operator- (const Mat3 & w) const { return Mat3(v[0]-w[0], v[1]-w[1], v[2]-w[2]); }
  Mat3 & operator-= (const Mat3 & w) { v[0] -= w[0]; v[1] -= w[1]; v[2] -= w[2]; return *this; }

  Mat3 operator* (real s) const { return Mat3(v[0]*s, v[1]*s, v[2]*s); }
  Mat3 & operator*= (real s) { v[0] *= s; v[1] *= s; v[2] *= s; return *this; }

  Mat3 operator/ (real s) const { return Mat3(v[0]/s, v[1]/s, v[2]/s); }
  Mat3 & operator/= (real s) { v[0] /= s; v[1] /= s; v[2] /= s; return *this; }

  real maxAbsEntry() const;

  Vec3<real> & operator[] (int index) { return v[index]; } // M[i] returns i-th row
  const Vec3<real> & operator[] (int index) const { return v[index]; } // M[i] returns i-th row

  Vec3<real> col(int index) const { return Vec3<real>(v[0][index], v[1][index], v[2][index]); } // get i-th column
  Vec3<real> diag() const { return Vec3<real>(v[0][0], v[1][1], v[2][2]); }

  // converts the matrix into a C array of length 9 in row-major order
  void convertToArray(real array[9]) const { for(int i = 0; i < 9; i++) array[i] = this->data()[i]; }
  void addToArray(real array[9]) const { for(int i = 0; i < 9; i++) array[i] += this->data()[i]; }
  void addToArray(real vec0[3], real vec1[3], real vec2[3]) const   { v[0].addToArray(vec0); v[1].addToArray(vec1); v[2].addToArray(vec2); }

  const real * data() const { return &v[0][0]; }
  real * data() { return &v[0][0]; }

  // matrix-vector multiply
  const Vec3<real> operator* (const Vec3<real> & vec) const { return(Vec3<real>( dot(v[0],vec), dot(v[1],vec), dot(v[2],vec))); }

  // matrix-matrix multiply
  const Mat3 operator* (const Mat3 & mat2) const;

  inline friend Mat3 operator- (const Mat3 & v) { return real(-1) * v; }
  inline friend Mat3 trans(const Mat3 & v) { return Mat3( v[0][0], v[1][0], v[2][0], v[0][1], v[1][1], v[2][1], v[0][2], v[1][2], v[2][2] ); }
  inline friend Mat3 operator* (real s, const Mat3 & v) { return Mat3(s*v[0], s*v[1], s*v[2]); }

  // compute [  0  -v2  v1 ]
  //         [  v2  0  -v0 ]
  //         [ -v1  v0  0  ]
  inline friend Mat3 skewSymmetricMatrix(const Vec3<real> & vec) { return Mat3<real>(0, -vec[2], vec[1], vec[2], 0, -vec[0], -vec[1], vec[0], 0); }

  // given two transforms f1(p) = R1 p + t1 and f2(p) = R2 p + t2,
  // compute the composite transform f1(f2(p)) = R1 R2 p + (R1 t2 + t1)
  // So Rout = R1 R2, tout = R1 t2 + t1
  // This is the same as using 4x4 traform matrices M1 = [R1 t1; 0 1] M2 = [R2 t2; 0 1]
  // and computing [Rout, tout; 0 1] = M1 M2
  inline friend void multiplyAffineTransform4ds(const Mat3<real> & R1, const Vec3<real> & t1, const Mat3<real> & R2, const Vec3<real> & t2,
      Mat3<real> & Rout, Vec3<real> & tout) { Rout = R1 * R2; tout = R1 * t2 + t1; }

  // return element rotation matrix that rotates around X, Y or Z axis
  //      [ 1  0  0]       [ c  0  s]       [ c -s  0]
  // Rx = [ 0  c -s]  Ry = [ 0  1  0]  Rz = [ s  c  0]
  //      [ 0  s  c]       [-s  0  c]       [ 0  0  1]
  // where c = cos(angle), s = sin(angle)
  static Mat3<real> getElementRotationMatrix(int axis, real angle);

protected:
  Vec3<real> v[3]; // the three rows of the matrix
};

template<typename real>
inline Mat3<real> tensorProduct(const Vec3<real> & vec1, const Vec3<real> & vec2);

template<typename real>
inline Mat3<real> inv(const Mat3<real> &); // inverse matrix

template<typename real>
inline real det(const Mat3<real> & mat); // determinant

// ===== below is the implementation =====

template<typename real>
inline Mat3<real> tensorProduct(const Vec3<real> & vecA, const Vec3<real> & vecB)
{
  return Mat3<real>(vecA[0]*vecB[0],vecA[0]*vecB[1],vecA[0]*vecB[2],
                    vecA[1]*vecB[0],vecA[1]*vecB[1],vecA[1]*vecB[2],
                    vecA[2]*vecB[0],vecA[2]*vecB[1],vecA[2]*vecB[2]);
}

template<typename real>
inline const Mat3<real> Mat3<real>::operator* (const Mat3<real> & mat2) const
{
  return Mat3<real>(
    dot(v[0],mat2.col(0)), dot(v[0],mat2.col(1)), dot(v[0],mat2.col(2)),
    dot(v[1],mat2.col(0)), dot(v[1],mat2.col(1)), dot(v[1],mat2.col(2)),
    dot(v[2],mat2.col(0)), dot(v[2],mat2.col(1)), dot(v[2],mat2.col(2)));
}

template<typename real>
inline Mat3<real> inv(const Mat3<real> & mat)
{
  Mat3<real> AInvMat;
  real * AInv = AInvMat.data();
  real * A = mat.data();

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

  real invDet = 1.0 / (A[0] * AInv[0] + A[1] * AInv[3] + A[2] * AInv[6]);

  for(int i=0; i<9; i++)
    AInv[i] *= invDet;

  return AInvMat;
}

template<typename real>
inline real det(const Mat3<real> & mat)
{
  return
   (-mat[0][2] * mat[1][1] * mat[2][0] +
     mat[0][1] * mat[1][2] * mat[2][0] +
     mat[0][2] * mat[1][0] * mat[2][1] -
     mat[0][0] * mat[1][2] * mat[2][1] -
     mat[0][1] * mat[1][0] * mat[2][2] +
     mat[0][0] * mat[1][1] * mat[2][2] );
}

template<typename real>
inline real Mat3<real>::maxAbsEntry() const
{
  real maxAbsEntry = 0.0;
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
    {
       if (fabs(v[i][j]) > maxAbsEntry)
         maxAbsEntry = fabs(v[i][j]);
    }
  return maxAbsEntry;
}

//      [ 1  0  0]       [ c  0  s]       [ c -s  0]
// Rx = [ 0  c -s]  Ry = [ 0  1  0]  Rz = [ s  c  0]
//      [ 0  s  c]       [-s  0  c]       [ 0  0  1]
template<typename real>
inline Mat3<real> Mat3<real>::getElementRotationMatrix(int axis, real angle) // axis is 0,1,2, angle in radian
{
  real c = cos(angle), s = sin(angle);
  real ns = -s;
  switch(axis)
  {
    case 0:
      return Mat3<real>(real(1), real(0), real(0), real(0), c, ns, real(0), s, c);
    case 1:
      return Mat3<real>(c, real(0), s, real(0), real(1), real(0), ns, real(0), c);
    case 2:
      return Mat3<real>(c, ns, real(0), s, c, real(0), real(0), real(0), real(1));
  }
  // invalid case
  return Mat3<real>(0.0);
}

#endif
