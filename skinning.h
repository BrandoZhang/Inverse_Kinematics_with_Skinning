#ifndef SKINNING_H
#define SKINNING_H

#include "transform4d.h"
#include <vector>
#include <string>

// A class to perform linear blend skinning on a triangle mesh.

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

class Skinning
{
public:
  // Load skinning data from a file.
  // numMeshVertices, restMeshVertexPositions: specifies the mesh vertices to be skinned
  // restMeshVertexPositions must be an array of length 3*numMeshVertices .
  // meshSkinningWeightsFilename: ASCII file in SparseMatrix format, giving the skinning weights.
  Skinning(int numMeshVertices, const double * restMeshVertexPositions, const std::string & meshSkinningWeightsFilename);

  // Main routine: Apply skinning to produce the new positions of the mesh vertices.
  // jointSkinTransforms is an array of transformations, one per joint. For each joint, we have: 
  // jointSkinTransform = globalTransform * globalRestTransform^{-1}
  // input: jointSkinTransforms
  // output: newMeshVertexPositions (length is 3*numMeshVertices)
  void applySkinning(const RigidTransform4d * jointSkinTransforms, double * newMeshVertexPositions) const;

protected:
  int numMeshVertices = 0;
  const double * restMeshVertexPositions = nullptr; // length of array is 3 x numMeshVertices

  // Number of joints that influence each vertex. This is constant for all vertices.
  int numJointsInfluencingEachVertex = 0; 
  // The indices of joints that affect each mesh vertex.
  // Length is: numJointsInfluencingEachVertex * numMeshVertices.
  std::vector<int> meshSkinningJoints;  
  // The skinning weights for each mesh vertex.
  // Length is numJointsInfluencingEachVertex * numMeshVertices.
  std::vector<double> meshSkinningWeights; 
};

#endif

