#include "skinning.h"
#include "vec3d.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

Skinning::Skinning(int numMeshVertices, const double * restMeshVertexPositions,
    const std::string & meshSkinningWeightsFilename)
{
  this->numMeshVertices = numMeshVertices;
  this->restMeshVertexPositions = restMeshVertexPositions;

  cout << "Loading skinning weights..." << endl;
  ifstream fin(meshSkinningWeightsFilename.c_str());
  assert(fin);
  int numWeightMatrixRows = 0, numWeightMatrixCols = 0;
  fin >> numWeightMatrixRows >> numWeightMatrixCols;
  assert(fin.fail() == false);
  assert(numWeightMatrixRows == numMeshVertices);
  int numJoints = numWeightMatrixCols;

  vector<vector<int>> weightMatrixColumnIndices(numWeightMatrixRows);
  vector<vector<double>> weightMatrixEntries(numWeightMatrixRows);
  fin >> ws;
  while(fin.eof() == false)
  {
    int rowID = 0, colID = 0;
    double w = 0.0;
    fin >> rowID >> colID >> w;
    weightMatrixColumnIndices[rowID].push_back(colID);
    weightMatrixEntries[rowID].push_back(w);
    assert(fin.fail() == false);
    fin >> ws;
  }
  fin.close();

  // Build skinning joints and weights.
  numJointsInfluencingEachVertex = 0;
  for (int i = 0; i < numMeshVertices; i++)
    numJointsInfluencingEachVertex = std::max(numJointsInfluencingEachVertex, (int)weightMatrixEntries[i].size());
  assert(numJointsInfluencingEachVertex >= 2);

  // Copy skinning weights from SparseMatrix into meshSkinningJoints and meshSkinningWeights.
  meshSkinningJoints.assign(numJointsInfluencingEachVertex * numMeshVertices, 0);
  meshSkinningWeights.assign(numJointsInfluencingEachVertex * numMeshVertices, 0.0);
  for (int vtxID = 0; vtxID < numMeshVertices; vtxID++)
  {
    vector<pair<double, int>> sortBuffer(numJointsInfluencingEachVertex);
    for (size_t j = 0; j < weightMatrixEntries[vtxID].size(); j++)
    {
      int frameID = weightMatrixColumnIndices[vtxID][j];
      double weight = weightMatrixEntries[vtxID][j];
      sortBuffer[j] = make_pair(weight, frameID);
    }
    sortBuffer.resize(weightMatrixEntries[vtxID].size());
    assert(sortBuffer.size() > 0);
    sort(sortBuffer.rbegin(), sortBuffer.rend()); // sort in descending order using reverse_iterators
    for(size_t i = 0; i < sortBuffer.size(); i++)
    {
      meshSkinningJoints[vtxID * numJointsInfluencingEachVertex + i] = sortBuffer[i].second;
      meshSkinningWeights[vtxID * numJointsInfluencingEachVertex + i] = sortBuffer[i].first;
    }

    // Note: When the number of joints used on this vertex is smaller than numJointsInfluencingEachVertex,
    // the remaining empty entries are initialized to zero due to vector::assign(XX, 0.0) .
  }
}

void Skinning::applySkinning(const RigidTransform4d * jointSkinTransforms, double * newMeshVertexPositions) const
{
  /* Linear Blend Skinning (LBS) */
  if (this->appliedSkinningMethod == LinearBlendSkinning)
  {
    for (int i = 0; i < numMeshVertices; i++)
    {
      Vec4d newVertexPosition;
      Vec4d restVertexPosition = Vec4d(restMeshVertexPositions[3 * i + 0],
                                       restMeshVertexPositions[3 * i + 1],
                                       restMeshVertexPositions[3 * i + 2],
                                       1.0);
      Mat4d accumulativeTransform;
      accumulativeTransform = Mat4d::Zero;
      for (int j = 0; j < numJointsInfluencingEachVertex; j++)
      {
        int currIdx = i * numJointsInfluencingEachVertex + j;
        accumulativeTransform += meshSkinningWeights[currIdx] * jointSkinTransforms[meshSkinningJoints[currIdx]];
      }
      newVertexPosition = accumulativeTransform * restVertexPosition;

      newMeshVertexPositions[3 * i + 0] = newVertexPosition[0];
      newMeshVertexPositions[3 * i + 1] = newVertexPosition[1];
      newMeshVertexPositions[3 * i + 2] = newVertexPosition[2];
    }
  }

  /* Dual Quaternion Skinning */
  else if (this->appliedSkinningMethod == DualQuaternionSkinning)
  {
    for (int i = 0; i < numMeshVertices; i++)
    {
      Eigen::Quaterniond qRestVertexPosition(0,
                                             restMeshVertexPositions[3 * i + 0],
                                             restMeshVertexPositions[3 * i + 1],
                                             restMeshVertexPositions[3 * i + 2]);
      Eigen::Quaterniond q0Blend(0, 0, 0, 0);
      Eigen::Quaterniond q1Blend(0, 0, 0, 0);
      for (int j = 0; j < numJointsInfluencingEachVertex; j++)
      {
        int currIdx = i * numJointsInfluencingEachVertex + j;
        Mat3d matRotation = jointSkinTransforms[meshSkinningJoints[currIdx]].getRotation();
        Vec3d vecTranslation = jointSkinTransforms[meshSkinningJoints[currIdx]].getTranslation();
        // Convert to Eigen type
        Eigen::Matrix3d EigenMatRotation;
        for (int rowID = 0; rowID < 3; rowID++)
          for (int colID = 0; colID < 3; colID++)
            EigenMatRotation(rowID, colID) = matRotation[rowID][colID];
        // Construct dual quaternion
        Eigen::Quaterniond q0(EigenMatRotation);
        if (q0.dot(q0Blend) < 0)  // q0 and q0Blend are not in the same hemisphere, flip q0
            q0.coeffs() = -q0.coeffs();
        Eigen::Quaterniond translation(0, vecTranslation[0], vecTranslation[1], vecTranslation[2]);
        Eigen::Quaterniond dqTemp = translation * q0;
        // Weighted sum
        q0Blend.coeffs() += meshSkinningWeights[currIdx] * q0.coeffs();
        q1Blend.coeffs() += 0.5 * meshSkinningWeights[currIdx] * dqTemp.coeffs();
      }
      Eigen::Quaterniond q0Tilde, q1Tilde;
      double q0BlendNorm = q0Blend.norm();
      q0Tilde = q0Blend.coeffs() / q0BlendNorm;
      q1Tilde = q1Blend.coeffs() / q0BlendNorm -
                (q0Blend.dot(q1Blend)) / pow(q0BlendNorm, 3) * q0Blend.coeffs();

      Eigen::Vector3d finalTranslation = 2.0 * (q1Tilde * q0Tilde.conjugate()).vec();
      Eigen::Vector3d newVertexPosition = (q0Tilde * qRestVertexPosition * q0Tilde.conjugate()).vec() +
                                          finalTranslation;
      newMeshVertexPositions[3 * i + 0] = newVertexPosition[0];
      newMeshVertexPositions[3 * i + 1] = newVertexPosition[1];
      newMeshVertexPositions[3 * i + 2] = newVertexPosition[2];
    }
  }
  
  else {
    std::cerr << "Error: Unknown Skinning method: " << this->appliedSkinningMethod << std::endl;
    exit(1);
  }
}

