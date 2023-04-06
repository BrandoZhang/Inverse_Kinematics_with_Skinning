/*************************************************************************
 *                                                                       *
 * Vega FEM Simulation Library Version 4.0                               *
 *                                                                       *
 * "sceneObject" library , Copyright (C) 2007 CMU, 2009 MIT, 2018 USC    *
 * All rights reserved.                                                  *
 *                                                                       *
 * Code authors: Jernej Barbic, Daniel Schroeder                         *
 * http://www.jernejbarbic.com/vega                                      *
 *                                                                       *
 * Research: Jernej Barbic, Hongyi Xu, Yijing Li,                        *
 *           Danyong Zhao, Bohan Wang,                                   *
 *           Fun Shing Sin, Daniel Schroeder,                            *
 *           Doug L. James, Jovan Popovic                                *
 *                                                                       *
 * Funding: National Science Foundation, Link Foundation,                *
 *          Singapore-MIT GAMBIT Game Lab,                               *
 *          Zumberge Research and Innovation Fund at USC,                *
 *          Sloan Foundation, Okawa Foundation,                          *
 *          USC Annenberg Foundation                                     *
 *                                                                       *
 * This library is free software; you can redistribute it and/or         *
 * modify it under the terms of the BSD-style license that is            *
 * included with this library in the file LICENSE.txt                    *
 *                                                                       *
 * This library is distributed in the hope that it will be useful,       *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the file     *
 * LICENSE.TXT for more details.                                         *
 *                                                                       *
 *************************************************************************/

/*
  A general deformable scene object.
  See also sceneObject.h . 
*/

#ifndef _SCENEOBJECTDEFORMABLE_H_
#define _SCENEOBJECTDEFORMABLE_H_

#include "lighting.h"
#include "sceneObjectWithRestPosition.h"

class SceneObjectDeformable : public virtual SceneObjectWithRestPosition
{
public:
  SceneObjectDeformable(const char * filenameOBJ);
  SceneObjectDeformable(ObjMesh * objMesh, bool deepCopy = true);
  virtual ~SceneObjectDeformable();

  // sets the current dynamic vertex positions to the rest position + specified deformation
  void SetVertexDeformations(const double * u);
  void SetVertexDeformations(const float * u);

  // adds deformations to current dynamicPosition of the vertices
  void AddVertexDeformations(const double * u);

  void ResetDeformationToRest();

  inline void GetSingleVertexRestPosition(int vertex, double * x, double * y, double * z);
  inline void SetSingleVertexRestPosition(int vertex, double x, double y, double z);
  inline void GetSingleVertexPositionFromBuffer(int vertex, double * x, double * y, double * z);
  inline const Vec3d & GetSingleVertexPositionFromBuffer(int vertex) const;
  inline Vec3d GetSingleVertexRestPosition(int vertex) const;

  virtual void SetLighting(Lighting * lighting);

protected:

};

inline void SceneObjectDeformable::GetSingleVertexRestPosition(int vertex, double * x, double * y, double * z)
{
  *x = restPosition[3*vertex+0];
  *y = restPosition[3*vertex+1];
  *z = restPosition[3*vertex+2];
}

inline void SceneObjectDeformable::SetSingleVertexRestPosition(int vertex, double x, double y, double z)
{
  restPosition[3*vertex+0] = x;
  restPosition[3*vertex+1] = y;
  restPosition[3*vertex+2] = z;
}

inline void SceneObjectDeformable::GetSingleVertexPositionFromBuffer(int vertex, double * x, double * y, double * z)
{
  Vec3d pos = mesh->getPosition(vertex);
  *x = pos[0];
  *y = pos[1];
  *z = pos[2];
}

inline const Vec3d& SceneObjectDeformable::GetSingleVertexPositionFromBuffer(int vertex) const
{
  return mesh->getPosition(vertex);
}

inline Vec3d SceneObjectDeformable::GetSingleVertexRestPosition(int vertex) const
{
  return Vec3d(&restPosition[3*vertex]);
}

#endif

