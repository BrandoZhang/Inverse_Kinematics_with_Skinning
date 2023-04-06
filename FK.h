#ifndef FK_H
#define FK_H

#include <memory.h>
#include <vector>
#include "vec3d.h"
#include "transform4d.h"

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

// This file implements Forward Kinematics (FK).

// Comment on Euler angles:
// Any 3D orientation can be achieved by composing three elemental rotations.
// Each Euler angles describe one of the three elemental rotations.
// The three elemental rotations may be extrinsic (rotations around axes xyz of the world coordinate system, which are assumed to remain constant),
// or intrinsic (rotations around axes of a rotated coordinate system XYZ).
// Without considering the choice of the two conventions (extrinsic and intrinsic), there are 12 sequences of rotation axes, divided into two groups:
// Proper Euler angles (z-x-z, x-y-x, y-z-y, z-y-z, x-z-x, y-x-y)
// Tait–Bryan angles (x-y-z, y-z-x, z-x-y, x-z-y, z-y-x, y-x-z) (also called, yaw, pitch, and roll)
//
// Given Euler angles specified as XYZ (or any other combination), we can convert it to 3x3 rotation matrix as follows. For matrices
//      [ 1        ]        [ cy    sy ]        [ cz -sz   ]
//  X = [   cx -sx ]    Y = [     1    ]    Z = [ sz  cz   ], where ck is cos(Euler angle of axis k), sk is sin(Euler angle of axis k).
//      [   sx  cx ]        [ -sy   cy ]        [        1 ]
// Then, the rotation matrix equals R = Z * Y * X (matrix product). 
// For a point p, we have R p = ZYX p, where X, Y, Z are elemental rotations.
//
// So here, the format XYZ is interpretted as: extrinsically, we first transform point p by X, then by Y, finally by Z.
// These Euler angles are the same one as used in the computer animation class, CSCI 520 at the University of Southern California, 
// lectured by Prof. Jernej Barbic.
// The same Euler angles are also used in Autodesk Maya, with "Rotate Order" set to "xyz".
// Maya's computation is described in
// http://help.autodesk.com/view/MAYAUL/2018/ENU/?guid=__cpp_ref_class_m_transformation_matrix_html
// Note that Maya uses post-mulitplication of matrices, so inside Maya a vector is represented as a row-vector and
// rotation is represented as a row-vector multiplying a matrix.
enum RotateOrder
{
  XYZ = 0,
  YZX,
  ZXY,
  XZY,
  YXZ,
  ZYX
};
inline RotateOrder getDefaultRotateOrder() { return XYZ; }

// Forward kinematics of a joint hierarchy.
// This class follows the implementation conventions used in Autodesk Maya.
// For the provided examples, the hierarchy was exported from a Maya joint system.
class FK
{
public:

  // Load skeleton data from a file.
  // jointHierarchyFilename: ASCII file giving a list of indices a[i]; the parent of the i-th joint is the a[i]-th joint; the parent of the root is "-1"
  // skeletonConfigFilename: ASCII file storing the rest configuration of each joint (translate, rotate, jointOrient and rotateOrder attributes)
  //                         Each line in the file gives the values for a single attribute for all the joints.
  // This constructor sets jointRestTranslations, jointRestEulerAngles, jointOrientations, jointRotateOrders.
  FK(const std::string & jointHierarchyFilename, const std::string & skeletonConfigFilename);

  // Based on the current euler angles (jointEulerAngles) of all joints, compute current 
  // values (jointLocalTransforms, jointGlobalTransforms, jointSkinTransforms) of all the joints.
  // This is the main routine that computes forward kinematics.
  void computeJointTransforms();

  // Use these functions to read and modify the current joint angles:
  Vec3d * getJointEulerAngles() { return jointEulerAngles.data(); }
  // You can say the following function also for writing, not just reading.
  //  For example: jointEulerAngle(7) = Vec3d(10.0, 30.0, -45.0);
  Vec3d & jointEulerAngle(int jointID) { return jointEulerAngles[jointID]; } 
  // Resets all joint angles to their rest pose values:
  void resetToRestPose();

  // Joint hierarchy accessor functions:
  int getNumJoints() const { return numJoints; }
  int getJointParent(int jointID) const { return jointParents[jointID]; } // parent of the root is -1, by definition
  // The returned children are sorted (ascending order).
  const std::vector<int> & getJointChildren(int jointID) const { return jointChildren[jointID]; }
  // The returned joints are sorted (ascending order).
  std::vector<int> getJointDescendents(int jointID) const;
  // Get the joint that appears at position "index" in a linear joint update order. This order is established
  // in the constructor. When one traverses the hierarchy in this order, children are guaranteed to appear after
  // the parents. So, when performing forward kinematics, you can use this order to guarantee that
  // the transformations of parents are computed before the transformations of the children.
  int getJointUpdateOrder(int index) const { return jointUpdateOrder[index]; }
  
  // Get joint values in the rest pose:
  const Vec3d & getJointRestTranslation(int jointID) const { return jointRestTranslations[jointID]; }
  const Vec3d & getJointRestEulerAngles(int jointID) const { return jointRestEulerAngles[jointID]; }
  const Vec3d & getJointOrient(int jointID) const { return jointOrientations[jointID]; }
  RotateOrder getJointRotateOrder(int jointID) const { return jointRotateOrders[jointID]; }

  // Get joint values in the current pose:
  Vec3d getJointGlobalPosition(int jointID) const { return jointGlobalTransforms[jointID].getTranslation(); }
  const RigidTransform4d & getJointGlobalTransform(int jointID) const { return jointGlobalTransforms[jointID]; }
  const RigidTransform4d * getJointSkinTransforms() const { return jointSkinTransforms.data(); } // the transforms are used for skinning

protected:
  void buildJointChildren();

  // See comment in the implementation file.
  static void computeLocalAndGlobalTransforms(
    const std::vector<Vec3d> & translations, const std::vector<Vec3d> & eulerAngles,
    const std::vector<Vec3d> & jointOrientationEulerAngles, const std::vector<RotateOrder> & rotateOrders,
    const std::vector<int> jointParents, const std::vector<int> & jointUpdateOrder,
    std::vector<RigidTransform4d> & localTransforms, std::vector<RigidTransform4d> & globalTransforms);

  // See comment in the implementation file.
  static void computeSkinningTransforms(
    const std::vector<RigidTransform4d> & globalTransforms, 
    const std::vector<RigidTransform4d> & invRestGlobalTransforms,
    std::vector<RigidTransform4d> & skinTransforms);

  int numJoints = 0;
  std::vector<int> jointParents; // the parent of the root is -1
  std::vector<std::vector<int>> jointChildren; // They are sorted; ascending order.
  // jointUpdateOrder: an array of length numJoints
  // Stores an ordering from root -> leaves, so that a joint's parent is always before the joint in this order.
  std::vector<int> jointUpdateOrder;

  // Translation, Euler angles, and orientation of the joints in the rest configuration.
  std::vector<Vec3d> jointRestTranslations, jointRestEulerAngles, jointOrientations;
  // Rotate orders of the joints. Different joints may have different rotation orders.
  std::vector<RotateOrder> jointRotateOrders;

  // Current values of various joint quantities:
  std::vector<Vec3d> jointEulerAngles; 
  std::vector<RigidTransform4d> jointLocalTransforms, jointGlobalTransforms, jointSkinTransforms;

  // jointInvRestGlobalTransforms are the inverse of restGlobalTransforms.
  // restGlobalTransforms are 4x4 row-major transforms. Same convention as in Maya (worldMatrix attribute).
  // JointInvRestGlobalTransform is the global matrix that transfers the coordinate of a point expressed in 
  // a local joint coordinate frame, to the world coordinate frame; when the joint hierarchy is in the rest pose.
  std::vector<RigidTransform4d> jointInvRestGlobalTransforms;
};

#endif

