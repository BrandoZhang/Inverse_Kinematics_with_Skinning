#include "FK.h"
#include "transform4d.h"
#include "graphSearchAlgorithms.h"
#include "basicAlgorithms.h"
#include "containerHelper.h"
#include "listIO.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <functional>
#if defined(_WIN32) || defined(WIN32)
  #ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
  #endif
#endif
#include <math.h>
using namespace std;

namespace
{

inline double deg2rad(double deg) { return deg * M_PI / 180.0; }

// Convert Euler angles (in degrees) in the given RotateOrder, to the row-major 3x3 rotation.
void euler2Rotation(const double angle[3], double R[9], RotateOrder order)
{
  Mat3d RX = getElementRotationMatrix(0, deg2rad(angle[0]));
  Mat3d RY = getElementRotationMatrix(1, deg2rad(angle[1]));
  Mat3d RZ = getElementRotationMatrix(2, deg2rad(angle[2]));

  Mat3d & RMat = asMat3d(R);
  switch(order)
  {
  case XYZ:
    RMat = RZ * RY * RX;
    return;
  case YZX:
    RMat = RX * RZ * RY;
    return;
  case ZXY:
    RMat = RY * RX * RZ;
    return;
  case XZY:
    RMat = RY * RZ * RX;
    return;
  case YXZ:
    RMat = RZ * RX * RY;
    return;
  case ZYX:
    RMat = RX * RY * RZ;
    return;
  }
  assert(0);
}

} // anonymous namespace

FK::FK(const std::string & jointParentsFilename, const std::string & skeletonConfigFilename)
{
  const int listOffset = 0;
  const bool sortListAfterLoad = false;
  int exitCode = ListIO::load(jointParentsFilename.c_str(), jointParents, listOffset, sortListAfterLoad);
  assert(exitCode == 0);
  numJoints = jointParents.size();
  cout << "Load # joints " << numJoints << " : " << streamRange(jointParents) << endl;

  ifstream fin(skeletonConfigFilename.c_str());
  assert(fin);
  // jointRestTransformConfig file format:
  //
  // <joint 0 translation vector> <joint 1 translation vector> ... 
  // <joint 0 rotation vector> <joint 1 rotation vector> ... 
  // <joint 0 jointOrientation vector> <joint 1 jointOrientation vector> ... 
  // (optional) <joint 0 rotateOrder ("xyz" or other orders) > <joint 0 rotateOrder> ... 
  
  // first, load translation, rotation and jointOrientation data
  jointRestTranslations.resize(numJoints);
  jointRestEulerAngles.resize(numJoints);
  jointOrientations.resize(numJoints);
  jointInvRestGlobalTransforms.resize(numJoints);

  for(int i = 0; i < numJoints; i++)
    fin >> jointRestTranslations[i][0] >> jointRestTranslations[i][1] >> jointRestTranslations[i][2];
  for(int i = 0; i < numJoints; i++)
    fin >> jointRestEulerAngles[i][0] >> jointRestEulerAngles[i][1] >> jointRestEulerAngles[i][2];
  for(int i = 0; i < numJoints; i++)
    fin >> jointOrientations[i][0] >> jointOrientations[i][1] >> jointOrientations[i][2];

  assert(fin.fail() == false);
  fin >> ws;

  jointEulerAngles = jointRestEulerAngles;

  jointRotateOrders.resize(numJoints, getDefaultRotateOrder());
  if (fin.eof() == false) // load rotateOrder
  {
    // {0: "XYZ", 1: "YZX", 2: "ZXY", 3: "XZY", 4: "YXZ", 5: "ZYX"}
    const map<string, RotateOrder> orderStrMap =
    {
      {"xyz", RotateOrder::XYZ},
      {"yzx", RotateOrder::YZX},
      {"zxy", RotateOrder::ZXY},
      {"xzy", RotateOrder::XZY},
      {"yxz", RotateOrder::YXZ},
      {"zyx", RotateOrder::ZYX}
    };
    string orderStr(3, '\0');
    for(int i = 0; i < numJoints; i++)
    {
      memset(&orderStr[0], 0, sizeof(char) * 3);
      for(int d = 0; d < 3; d++)
        fin.get(orderStr[d]);
      for(int d = 0; d < 3; d++)
        orderStr[d] = tolower(orderStr[d]);
//      cout << "joint " << i << " " << orderStr << endl;
      auto it = orderStrMap.find(orderStr);
      assert(it != orderStrMap.end());
      jointRotateOrders[i] = it->second;
      fin >> ws;
    }
    assert(fin.fail() == false);
  }
  fin.close();

  vector<bool> jointVisited(numJoints, false);
  // Use recursion to create an order to update joint transforms, 
  // so that one joint's parent always gets updated before the joint.
  std::function<void(int)> goToParent = [&](int jointID) -> void
  {
    if (jointVisited[jointID])
      return;
    if (jointParents[jointID] >= 0) // this joint is not the root
    {
      goToParent(jointParents[jointID]);
    }

    jointUpdateOrder.push_back(jointID);
    jointVisited[jointID] = true;
  };

  jointUpdateOrder.clear();
  for(int i = 0; i < numJoints; i++)
    goToParent(i);
  assert(jointUpdateOrder.size() == (size_t)numJoints);
  for(int i = 0; i < numJoints; i++)
    assert(jointParents[i] < 0 || jointUpdateOrder[jointParents[i]] < jointUpdateOrder[i]);

  // Call computeLocalAndGlobalTransforms to compute jointRestGlobalTransforms given restTranslations/EulerAngles/JointOrientations.
  // jointInvRestGlobalTransforms here is just a place-holder.
  vector<RigidTransform4d> jointRestGlobalTransforms(numJoints);
  computeLocalAndGlobalTransforms(jointRestTranslations, jointRestEulerAngles, jointOrientations, jointRotateOrders, 
      jointParents, jointUpdateOrder,
      jointInvRestGlobalTransforms/*not used*/, jointRestGlobalTransforms);

  for (int i = 0; i < numJoints; i++)
  {
    jointInvRestGlobalTransforms[i] = inv(jointRestGlobalTransforms[i]);
  }

  buildJointChildren();

  jointLocalTransforms.resize(numJoints);
  jointGlobalTransforms.resize(numJoints);
  jointSkinTransforms.resize(numJoints);
  computeJointTransforms();
}

void FK::buildJointChildren()
{
  jointChildren.assign(numJoints, {});
  for(int jointID = 0; jointID < numJoints; jointID++)
  {
    int parentID = jointParents[jointID];
    if (parentID < 0)
      continue;
    jointChildren[parentID].push_back(jointID);
  }
}

vector<int> FK::getJointDescendents(int jointID) const
{
  vector<int> ret = jointChildren[jointID];
  doBFSOnDirectedTree(convertArrayToFunction(jointChildren), ret);
  sort(ret.begin(), ret.end());
  return ret;
}

// This is the main function that performs forward kinematics.
// Each joint has its local transformation relative to the parent joint. 
// globalTransform of a joint is the transformation that converts a point expressed in the joint's local frame of reference, to the world coordinate frame.
// localTransform is the transformation that converts a point expressed in the joint's local frame of reference, to the coordinate frame of its parent joint. Specifically, if xLocal is the homogeneous coordinate of a point expressed in the joint's local frame, and xParent is the homogeneous coordinate of the same point expressed in the frame of the joint's parent, we have:
// xParent = localTransform * xLocal , and
// globalTransform = parentGlobalTransform * localTransform .
// Note that the globalTransform of the root joint equals its localTransform.
//
// Input: translations of each joint relative to its parent (in parent's coordinate system), 
// current Euler angles, joint orientations, joint rotation order, joint parents, joint update order.
// All arrays are assumed to have the same length (= #joints).
// Joint orientations are always given in XYZ order (Maya convention).
// Output: localTransforms and globalTransforms.
void FK::computeLocalAndGlobalTransforms(
    const vector<Vec3d> & translations, const vector<Vec3d> & eulerAngles, 
    const vector<Vec3d> & jointOrientationEulerAngles, const vector<RotateOrder> & rotateOrders,
    const std::vector<int> jointParents, const vector<int> & jointUpdateOrder,
    vector<RigidTransform4d> & localTransforms, vector<RigidTransform4d> & globalTransforms)
{
  // Students should implement this.
  // First, compute the localTransform for each joint, using eulerAngles and jointOrientationEulerAngles,
  // and the "euler2Rotation" function.
  // Then, recursively compute the globalTransforms, from the root to the leaves of the hierarchy.
  // Use the jointParents and jointUpdateOrder arrays to do so.
  // Also useful are the Mat3d and RigidTransform4d classes defined in the Vega folder.

  // The following is just a dummy implementation that should be replaced.
  double identity[16] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1 };
  for(int i=0; i<localTransforms.size(); i++)
  {
    localTransforms[i] = RigidTransform4d(identity);
    globalTransforms[i] = RigidTransform4d(identity);
  }
}

// Compute skinning transformations for all the joints, using the formula:
// skinTransform = globalTransform * invRestTransform
void FK::computeSkinningTransforms(
    const vector<RigidTransform4d> & globalTransforms, 
    const vector<RigidTransform4d> & invRestGlobalTransforms,
    vector<RigidTransform4d> & skinTransforms)
{
  // Students should implement this.

  // The following is just a dummy implementation that should be replaced.
  double identity[16] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1 };
  for(int i=0; i<skinTransforms.size(); i++)
  {
    skinTransforms[i] = RigidTransform4d(identity);
  }
}

void FK::computeJointTransforms()
{
  computeLocalAndGlobalTransforms(jointRestTranslations, jointEulerAngles, jointOrientations, jointRotateOrders, 
      jointParents, jointUpdateOrder,
      jointLocalTransforms, jointGlobalTransforms);
  computeSkinningTransforms(jointGlobalTransforms, jointInvRestGlobalTransforms, jointSkinTransforms);
}

void FK::resetToRestPose()
{
  jointEulerAngles = jointRestEulerAngles;
  computeJointTransforms();
}

