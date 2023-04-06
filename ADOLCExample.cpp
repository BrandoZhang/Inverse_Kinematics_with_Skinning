#include <adolc/adolc.h>
#include <iostream>
#include <vector>
using namespace std;

// An example of how to use ADOL-C to evaluate functions, and Jacobian matrices (gradients) of functions.
// CSCI 520 Computer Animation and Simulation
// Jernej Barbic, Yijing Li

int main()
{
  // We define a function y = f(x), mapping R^3 --> R^2. Let y=[y0 y1]^T, and x=[x0 x1 x2]^T.
  // Then, f is defined as:
  // y0 = x0 + 2 x2 + 3 x2
  // y1 = 4 x0 + 5 x1
  // Our goal is to use ADOL-C to compute the Jacobian matrix of f:
  // J = 
  // [1 2 3]
  // [4 5 0]
  int n = 3; // input dimension is n
  int m = 2; // output dimension is m

  // first, call trace_on to ask ADOL-C to begin recording how function f is implemented
  int adolc_tagID = 1; // This is an ID used in ADOL-C to represent each individual function.
  trace_on(adolc_tagID); // start tracking computation with ADOL-C

  // ADOL-C uses its own double implementation: adouble to track floating-point operations used in f
  // adouble overloads basic C++ arithmetic operations like =,+,-,*,/ so that ADOL-C knows when and where
  // adoubles are involved in which operations.
  // Note: since adoubles are not real doubles, they have complicated data structures inside. So
  // you shouldn't use direct memory functions like memcpy((adouble*)..., ..., ...) to edit adoubles.
  vector<adouble> x(n); // define the input of the function f
  for(int i = 0; i < n; i++)
    x[i] <<= 0.0; // The <<= syntax tells ADOL-C that these are the input variables.

  vector<adouble> y(m); // define the output of the function f

  // The computation of f goes here:
  y[0] = x[0] + 2 * x[1] + 3 * x[2];
  y[1] = 4 * x[0] + 5 * x[1];

  vector<double> output(m);
  for(int i = 0; i < m; i++)
    y[i] >>= output[i]; // Use >>= to tell ADOL-C that y[i] are the output variables

  // Finally, call trace_off to stop recording the function f.
  trace_off(); // ADOL-C tracking finished

  // now, you can call ::function(adolc_tagID, ...) as many times as you like to ask ADOL-C to evaluate f for different x:
  double input_x_values[] = {0.0, 0.1, 0.2};
  double output_y_values[] = {0.0, 0.0};
  ::function(adolc_tagID, m, n, input_x_values, output_y_values);

  cout << "output_y_values: " << output_y_values[0] << " " << output_y_values[1] << endl;

  // You can call ::jacobian(adolc_tagID, ...) as many times as you like to ask ADOL-C to evalute the jacobian matrix of f on different x:
  double jacobianMatrix[2*3]; // We store the matrix in row-major order.
  double * jacobianMatrixEachRow[] = { &jacobianMatrix[0], &jacobianMatrix[3] }; // pointer array where each pointer points to one row of the jacobian matrix
  ::jacobian(adolc_tagID, m, n, input_x_values, jacobianMatrixEachRow); // each row is the gradient of one output component of the function

  cout << "jacobian matrix: " << endl;
  for(int i = 0; i < 2; i++)
  {
    for(int j = 0; j < 3; j++)
      cout << jacobianMatrix[i*3+j] << " ";
    cout << endl;
  }

  return 0;
}

