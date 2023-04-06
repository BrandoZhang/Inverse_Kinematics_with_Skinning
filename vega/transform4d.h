#ifndef AFFINETRANSFORM4D_H
#define AFFINETRANSFORM4D_H

#include "mat4d.h"
#include "mat3d.h"

// a 4x4 row-major matrix representing an affine transformation
// [ A t ]
// [ 0 1 ]
class AffineTransform4d : public Mat4d
{
public:
  // initialize to be an identity transformation
  inline AffineTransform4d() : Mat4d(Mat4d::Identity) {}
  inline AffineTransform4d(const double mat[16]) : Mat4d(mat) {}
  inline AffineTransform4d(const Mat4d & mat) : Mat4d(mat) {}
  inline AffineTransform4d(const Mat3d & A, const Vec3d & t);

  inline Mat3d getLinearTrans() const; // get row-major 3x3 matrix representing the linear transformation part
  inline Vec3d getTranslation() const;
  inline Vec3d transformPoint(const Vec3d & p) const;
  inline Vec3d transformVector(const Vec3d & v) const;
};

// a 4x4 row-major matrix representing a rigid transformation
// [ R t ]
// [ 0 1 ]
class RigidTransform4d : public AffineTransform4d
{
public:
  inline RigidTransform4d() : AffineTransform4d() {}
  inline RigidTransform4d(const double mat[16]) : AffineTransform4d(mat) {}
  inline RigidTransform4d(const Mat4d & mat) : AffineTransform4d(mat) {}
  inline RigidTransform4d(const AffineTransform4d & mat) : AffineTransform4d(mat) {}
  inline RigidTransform4d(const Mat3d & A, const Vec3d & t) : AffineTransform4d(A, t) {}

  inline Mat3d getRotation() const { return AffineTransform4d::getLinearTrans(); }
};

inline RigidTransform4d inv(const RigidTransform4d & mat) // inverse matrix
{
  Mat3d RT = trans(mat.getRotation());
  return RigidTransform4d(RT, - (RT * mat.getTranslation()));
}

// =============== IMPLEMENTATION ===============

inline AffineTransform4d::AffineTransform4d(const Mat3d & A, const Vec3d & t)
{
  A[0].convertToArray(elt[0].data());
  A[1].convertToArray(elt[1].data());
  A[2].convertToArray(elt[2].data());
  elt[0][3] = t[0];
  elt[1][3] = t[1];
  elt[2][3] = t[2];
  elt[3] = Vec4d(0,0,0,1);
}

inline Mat3d AffineTransform4d::getLinearTrans() const
{
  return Mat3d(elt[0].data(), elt[1].data(), elt[2].data());
}

inline Vec3d AffineTransform4d::getTranslation() const
{
  return Vec3d(elt[0][3], elt[1][3], elt[2][3]);
}

inline Vec3d AffineTransform4d::transformPoint(const Vec3d & p) const
{
  return Vec3d (
    dot(p, asVec3d(elt[0].data())) + elt[0][3],
    dot(p, asVec3d(elt[1].data())) + elt[1][3],
    dot(p, asVec3d(elt[2].data())) + elt[2][3]
    );
}

inline Vec3d AffineTransform4d::transformVector(const Vec3d & p) const
{
  return Vec3d (
    dot(p, asVec3d(elt[0].data())),
    dot(p, asVec3d(elt[1].data())),
    dot(p, asVec3d(elt[2].data()))
    );
}

#endif
