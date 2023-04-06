#include <Eigen/Dense>
#include <iostream>
#include <vector>
using namespace std;

// A simple example on how to use Eigen to solve linear systems.
// CSCI 520 Computer Animation and Simulation
// Jernej Barbic, Yijing Li

int main()
{
  // a 3x3 system matrix A
  vector<double> sysMatrix =
  {
  	1, 2, 1,
  	2, 1, 0,
  	1, 0, 3
  };
  // system right-hand-side b
  vector<double> rhs = {0, 1, 0};

  // now we solve A x = b using Eigen

  // first, convert data to Eigen types
  Eigen::MatrixXd A(3,3); // define a 3x3 Eigen column-major matrix
  // assign values to A
  for(int rowID = 0; rowID < 3; rowID++)
    for(int colID = 0; colID < 3; colID++)
      A(rowID,colID) = sysMatrix[3*rowID+colID];

  Eigen::VectorXd b(3); // define a 3x1 Eigen column vector
  for(int i = 0; i < 3; i++)
    b(i) = rhs[i];

  // now solve for x in A x = b
  // note: here, we assume that A is symmetric; hence we can use the LDLT decomposition
  Eigen::VectorXd x = A.ldlt().solve(b);

  // check the accuracy of the solution by computing: ||Ax-b||_2
  double error = (A * x - b).norm();
  cout << "System solve error: " << error << endl;

  return 0;
}

