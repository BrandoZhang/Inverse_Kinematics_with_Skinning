#ifndef IK_H
#define IK_H

// Use adol-c to compute the gradient of the forward kinematics (the "Jacobian matrix"),
// then use Tikhonov regularization and the Jacobian matrix to perform IK.

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

#include <cfloat>

class FK;
class Vec3d;

class IK
{
public:
  // IK constructor.
  // numIKJoints, IKJointIDs: the number of IK handle joints, and their indices (using the joint numbering as defined in the FK class).
  // FK: pointer to an already initialized forward kinematics class.
  // adolc_tagID: an ID used in adol-c to represent a particular function for evaluation. Different functions should have different tagIDs.
  IK(int numIKJoints, const int * IKJointIDs, FK * fk, int adolc_tagID = 1);

  // input: an array of numIKJoints Vec3d's giving the positions of the IK handles, current joint Euler angles
  // output: the computed joint Euler angles; same meaning as in the FK class
  // Note: eulerAngles is both input and output
  void doIK(const Vec3d * targetHandlePositions, Vec3d * eulerAngles);

  // IK parameters
  int getFKInputDim() const { return FKInputDim; }
  int getFKOutputDim() const { return FKOutputDim; }
  int getIKInputDim() const { return FKOutputDim; }
  int getIKOutputDim() const { return FKInputDim; }

protected:
  int numIKJoints = 0;
  const int * IKJointIDs = nullptr;
  FK * fk = nullptr;
  int adolc_tagID = 0; // tagID
  int FKInputDim = 0; // forward dynamics input dimension 
  int FKOutputDim = 0; // forward dynamics output dimension

  void train_adolc();
};

#endif

