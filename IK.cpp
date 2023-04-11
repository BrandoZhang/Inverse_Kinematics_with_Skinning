#include "IK.h"
#include "FK.h"
#include "minivectorTemplate.h"
#include <Eigen/Dense>
#include <adolc/adolc.h>
#include <cassert>
#if defined(_WIN32) || defined(WIN32)
  #ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
  #endif
#endif
#include <math.h>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

namespace
{

// Converts degrees to radians.
template<typename real>
inline real deg2rad(real deg) { return deg * M_PI / 180.0; }

template<typename real>
Mat3<real> Euler2Rotation(const real angle[3], RotateOrder order)
{
  Mat3<real> RX = Mat3<real>::getElementRotationMatrix(0, deg2rad(angle[0]));
  Mat3<real> RY = Mat3<real>::getElementRotationMatrix(1, deg2rad(angle[1]));
  Mat3<real> RZ = Mat3<real>::getElementRotationMatrix(2, deg2rad(angle[2]));

  switch(order)
  {
    case RotateOrder::XYZ:
      return RZ * RY * RX;
    case RotateOrder::YZX:
      return RX * RZ * RY;
    case RotateOrder::ZXY:
      return RY * RX * RZ;
    case RotateOrder::XZY:
      return RY * RZ * RX;
    case RotateOrder::YXZ:
      return RZ * RX * RY;
    case RotateOrder::ZYX:
      return RX * RY * RZ;
  }
  assert(0);
}

// Performs forward kinematics, using the provided "fk" class.
// This is the function whose Jacobian matrix will be computed using adolc.
// numIKJoints and IKJointIDs specify which joints serve as handles for IK:
//   IKJointIDs is an array of integers of length "numIKJoints"
// Input: numIKJoints, IKJointIDs, fk, eulerAngles (of all joints)
// Output: handlePositions (world-coordinate positions of all the IK joints; length is 3 * numIKJoints)
template<typename real>
void forwardKinematicsFunction(
    int numIKJoints, const int * IKJointIDs, const FK & fk,
    const std::vector<real> & eulerAngles, std::vector<real> & handlePositions)
{
  // The implementation of this function is very similar to function computeLocalAndGlobalTransforms in the FK class.
  // The recommended approach is to first implement FK::computeLocalAndGlobalTransforms.
  // Then, implement the same algorithm into this function. To do so,
  // you can use fk.getJointUpdateOrder(), fk.getJointRestTranslation(), and fk.getJointRotateOrder() functions.
  // Also useful is the multiplyAffineTransform4ds function in minivectorTemplate.h .
  // It would be in principle possible to unify this "forwardKinematicsFunction" and FK::computeLocalAndGlobalTransforms(),
  // so that code is only written once. We considered this; but it is actually not easily doable.
  // If you find a good approach, feel free to document it in the README file, for extra credit.
    int n = fk.getNumJoints();
    vector<Mat3<real>> globalTransforms(n), localTransforms(n);
    vector<Vec3<real>> globalTranslations(n), localTranslations(n);
    for (int i = 0; i < n; i++)
    {
      int currentIdx = fk.getJointUpdateOrder(i);
      int parentIdx = fk.getJointParent(currentIdx);
      // Computes the local transformation matrix for joint "currentIdx"
      real angleEulerJointRotation[3] = {eulerAngles[3 * currentIdx + 0],
                                         eulerAngles[3 * currentIdx + 1],
                                         eulerAngles[3 * currentIdx + 2]};
      Mat3<real> matEulerJointRotation = Euler2Rotation(angleEulerJointRotation, fk.getJointRotateOrder(currentIdx));
      real angleJointOrientation[3] = {fk.getJointOrient(currentIdx)[0],
                                       fk.getJointOrient(currentIdx)[1],
                                       fk.getJointOrient(currentIdx)[2]};
      Mat3<real> matEulerJointOrientation = Euler2Rotation(angleJointOrientation, getDefaultRotateOrder());
      localTransforms[currentIdx] = matEulerJointOrientation * matEulerJointRotation;
      localTranslations[currentIdx] = {fk.getJointRestTranslation(currentIdx)[0],
                                       fk.getJointRestTranslation(currentIdx)[1],
                                       fk.getJointRestTranslation(currentIdx)[2]};

      if (parentIdx == -1)  // root joint
      {
        globalTransforms[currentIdx] = localTransforms[currentIdx];
        globalTranslations[currentIdx] = localTranslations[currentIdx];
      }
      else
      {
        multiplyAffineTransform4ds(globalTransforms[parentIdx],  // R1
                                   globalTranslations[parentIdx],  // t1
                                   localTransforms[currentIdx],  // R2
                                   localTranslations[currentIdx],  // t2
                                   globalTransforms[currentIdx],  // Rout
                                   globalTranslations[currentIdx]);  // tout
      }
    }
    // Computes handlePositions
    for (int i = 0; i < numIKJoints; i++)
    {
      int jointIdx = IKJointIDs[i];
      handlePositions[i * 3 + 0] = globalTranslations[jointIdx][0];
      handlePositions[i * 3 + 1] = globalTranslations[jointIdx][1];
      handlePositions[i * 3 + 2] = globalTranslations[jointIdx][2];
    }
}

} // end anonymous namespaces

IK::IK(int numIKJoints, const int * IKJointIDs, FK * inputFK, int adolc_tagID)
{
  this->numIKJoints = numIKJoints;
  this->IKJointIDs = IKJointIDs;
  this->fk = inputFK;
  this->adolc_tagID = adolc_tagID;

  FKInputDim = fk->getNumJoints() * 3;
  FKOutputDim = numIKJoints * 3;

  train_adolc();
}

void IK::train_adolc()
{
  // Here, you should setup adol_c:
  //   Define adol_c inputs and outputs. 
  //   Use the "forwardKinematicsFunction" as the function that will be computed by adol_c.
  //   This will later make it possible for you to compute the gradient of this function in IK::doIK
  //   (in other words, compute the "Jacobian matrix" J).
  // See ADOLCExample.cpp .
  int n = FKInputDim;  // input dimension
  int m = FKOutputDim;  // output dimension

  trace_on(adolc_tagID);  // start tracking computation with ADOL-C
    vector<adouble> x(n); // define the input of the function f
    for (int i = 0; i < n; i++)
      x[i] <<= 0.0; // The <<= syntax tells ADOL-C that these are the input variables.
    vector<adouble> y(m); // define the output of the function f
    forwardKinematicsFunction(numIKJoints, IKJointIDs, *fk, x, y);  // call the function f
    vector<double> output(m);
    for (int i = 0; i < m; i++)
      y[i] >>= output[i]; // Use >>= to tell ADOL-C that y[i] are the output variables
  trace_off();  // ADOL-C tracking finished
}

void IK::doIK(const Vec3d * targetHandlePositions, Vec3d * jointEulerAngles)
{
  // You may find the following helpful:
  int numJoints = fk->getNumJoints(); // Note that is NOT the same as numIKJoints!

  // Use adolc to evalute the forwardKinematicsFunction and its gradient (Jacobian). It was trained in train_adolc().
  // Specifically, use ::function, and ::jacobian .
  // See ADOLCExample.cpp .
  //
  // Use it implement the Tikhonov IK method (or the pseudoinverse method for extra credit).
  // Note that at entry, "jointEulerAngles" contains the input Euler angles. 
  // Upon exit, jointEulerAngles should contain the new Euler angles.

  /* Tikhonov IK Method */
  if (this->appliedIKMethod == Tikhonov)
  {
    /* To solve the linear system using Eigen's LDLT decomposition, we need to define A x = b:
     * where  x is the vector of delta joint angles to be solved for,
     *        A := J^T J + alpha I,
     *        b := J^T deltaPosition
     *  in Tikhonov IK Method. alpha is the regularization parameter.
     */
    vector<double> output_y_values(FKOutputDim);  // This is the output of the forwardKinematicsFunction.
    Eigen::VectorXd x(FKInputDim);  // The vector of delta joint angles to be solved for
    ::function(adolc_tagID, FKOutputDim, FKInputDim, jointEulerAngles->data(), output_y_values.data());

    // You can call ::jacobian(adolc_tagID, ...) as many times as you like to ask ADOL-C to evalute the jacobian matrix of f on different x:
    vector<double> jacobianMatrix(FKOutputDim * FKInputDim);  // We store the matrix in row-major order.
    vector<double*> jacobianMatrixEachRow(FKOutputDim);  // pointer array where each pointer points to one row of the jacobian matrix

    for (int i = 0; i < FKOutputDim; i++)  // Initialize the pointer array
      jacobianMatrixEachRow[i] = &jacobianMatrix[i * FKInputDim];
    ::jacobian(adolc_tagID, FKOutputDim, FKInputDim, jointEulerAngles->data(), jacobianMatrixEachRow.data());  // each row is the gradient of one output component of the function

    // First, convert data to Eigen types (to call ldlt() to solve the linear system)
    Eigen::MatrixXd J(FKOutputDim, FKInputDim);  // define a column-major matrix of FKOutputDim rows and FKInputDim columns
    for (int rowID = 0; rowID < FKOutputDim; rowID++)
      for (int colID = 0; colID < FKInputDim; colID++)
        J(rowID, colID) = jacobianMatrix[FKInputDim * rowID + colID];  // copy the jacobian matrix from row-major to column-major order
    Eigen::MatrixXd JT(FKInputDim, FKOutputDim);  // transpose of J
    JT = J.transpose();

    // Define A
    Eigen::MatrixXd I(FKInputDim, FKInputDim);
    I = Eigen::MatrixXd::Identity(FKInputDim, FKInputDim);
    Eigen::MatrixXd A(FKInputDim, FKInputDim);
    A = JT * J + this->alpha * I;
    // Define b
    Eigen::VectorXd b(FKInputDim), deltaP(FKOutputDim);
    for (int i = 0; i < FKOutputDim; i++)
	    deltaP[i] = targetHandlePositions->data()[i] - output_y_values.data()[i];
    b = JT * deltaP;
    // Solve for x
    // note: here, we assume that A is symmetric; hence we can use the LDLT decomposition
    x = A.ldlt().solve(b);

    // Update Joint Angles
    for (int i = 0; i < numJoints; i++)
    {
      jointEulerAngles[i] += Vec3d(x[3 * i + 0],
                                   x[3 * i + 1],
                                   x[3 * i + 2]);
    }
    return;
  }

  /* Pseudo Inverse Method */
  else if (this->appliedIKMethod == PseudoInverse)
  {
    /* The x can be calculated by:
     *    x = J_Dagger * deltaPosition
     * where  x is the vector of delta joint angles to be solved for,
     *        J_Dagger := J^T (J J^T)^(-1)  the pseudo inverse of J
     */
    vector<double> output_y_values(FKOutputDim);  // This is the output of the forwardKinematicsFunction.
    Eigen::VectorXd x(FKInputDim);  // The vector of delta joint angles to be solved for
    ::function(adolc_tagID, FKOutputDim, FKInputDim, jointEulerAngles->data(), output_y_values.data());

    // You can call ::jacobian(adolc_tagID, ...) as many times as you like to ask ADOL-C to evalute the jacobian matrix of f on different x:
    vector<double> jacobianMatrix(FKOutputDim * FKInputDim);  // We store the matrix in row-major order.
    vector<double*> jacobianMatrixEachRow(FKOutputDim);  // pointer array where each pointer points to one row of the jacobian matrix

    for (int i = 0; i < FKOutputDim; i++)  // Initialize the pointer array
      jacobianMatrixEachRow[i] = &jacobianMatrix[i * FKInputDim];
    ::jacobian(adolc_tagID, FKOutputDim, FKInputDim, jointEulerAngles->data(), jacobianMatrixEachRow.data());  // each row is the gradient of one output component of the function

    // First, convert data to Eigen types (to call ldlt() to solve the linear system)
    Eigen::MatrixXd J(FKOutputDim, FKInputDim);  // define a column-major matrix of FKOutputDim rows and FKInputDim columns
    for (int rowID = 0; rowID < FKOutputDim; rowID++)
      for (int colID = 0; colID < FKInputDim; colID++)
        J(rowID, colID) = jacobianMatrix[FKInputDim * rowID + colID];  // copy the jacobian matrix from row-major to column-major order
    Eigen::MatrixXd JT(FKInputDim, FKOutputDim);  // transpose of J
    JT = J.transpose();

    Eigen::MatrixXd J_Dagger(FKInputDim, FKOutputDim);
    J_Dagger = JT * (J * JT).inverse();
    Eigen::VectorXd deltaP(FKOutputDim);
    for (int i = 0; i < FKOutputDim; i++)
      deltaP[i] = targetHandlePositions->data()[i] - output_y_values.data()[i];
    x = J_Dagger * deltaP;

    // Update Joint Angles
    for (int i = 0; i < numJoints; i++)
    {
      jointEulerAngles[i] += Vec3d(x[3 * i + 0],
                                   x[3 * i + 1],
                                   x[3 * i + 2]);
    }
    return;
  }

  /* Tikhonov IK Method with Kernel Trick */
  else if (this->appliedIKMethod == KernelTrickOnTikhonov)
  {
    /* Similar to naive Tikhonov Method, this method also solve a linear system, but does not use LDLT decomposition.
     *    x = J^T (J J^T + alpha I)^{-1} deltaPosition
     * where  x is still the vector of delta joint angles to be solved for,
     *        alpha is the regularization parameter.
     * It can be proved that:
     *    x = (J^T J + alpha)^{-1} J^T deltaPosition = J^T (J J^T + alpha I)^{-1} deltaPosition
     * where the former one comes from directly solve the linear system (naive Tikhonov),
     * and the latter one is Tikhonov IK Method with Kernel Trick.
     * The trick here is, `FKOutputDim` is generally much less than `FKInputDim`.
     * Therefore, matrix multiplication of the latter one is much less than previous one.
     */
    vector<double> output_y_values(FKOutputDim);  // This is the output of the forwardKinematicsFunction.
    Eigen::VectorXd x(FKInputDim);  // The vector of delta joint angles to be solved for
    ::function(adolc_tagID, FKOutputDim, FKInputDim, jointEulerAngles->data(), output_y_values.data());

    // You can call ::jacobian(adolc_tagID, ...) as many times as you like to ask ADOL-C to evalute the jacobian matrix of f on different x:
    vector<double> jacobianMatrix(FKOutputDim * FKInputDim);  // We store the matrix in row-major order.
    vector<double*> jacobianMatrixEachRow(FKOutputDim);  // pointer array where each pointer points to one row of the jacobian matrix

    for (int i = 0; i < FKOutputDim; i++)  // Initialize the pointer array
      jacobianMatrixEachRow[i] = &jacobianMatrix[i * FKInputDim];
    ::jacobian(adolc_tagID, FKOutputDim, FKInputDim, jointEulerAngles->data(), jacobianMatrixEachRow.data());  // each row is the gradient of one output component of the function

    // First, convert data to Eigen types (to call ldlt() to solve the linear system)
    Eigen::MatrixXd J(FKOutputDim, FKInputDim);  // define a column-major matrix of FKOutputDim rows and FKInputDim columns
    for (int rowID = 0; rowID < FKOutputDim; rowID++)
      for (int colID = 0; colID < FKInputDim; colID++)
        J(rowID, colID) = jacobianMatrix[FKInputDim * rowID + colID];  // copy the jacobian matrix from row-major to column-major order
    Eigen::MatrixXd JT(FKInputDim, FKOutputDim);  // transpose of J
    JT = J.transpose();

    Eigen:: MatrixXd I(FKOutputDim, FKOutputDim);  // the size is expected to be smaller than that in naive Tikhonov
    I = Eigen::MatrixXd::Identity(FKOutputDim, FKOutputDim);
    Eigen::VectorXd deltaP(FKOutputDim);
    for (int i = 0; i < FKOutputDim; i++)
      deltaP[i] = targetHandlePositions->data()[i] - output_y_values.data()[i];
    x = JT * (J * JT + this->alpha * I).inverse() * deltaP;
    // Update Joint Angles
    for (int i = 0; i < numJoints; i++)
    {
      jointEulerAngles[i] += Vec3d(x[3 * i + 0],
                                   x[3 * i + 1],
                                   x[3 * i + 2]);
    }
    return;
  }

  else {
    std::cerr << "Error: Unknown IK method: " << this->appliedIKMethod << std::endl;
    exit(1);
  }
}

