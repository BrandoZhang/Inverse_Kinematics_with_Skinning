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
  A library to load and render an obj mesh (called here a "scene object").
  This library depends on the "objMesh" library. See also objMesh.h.
*/

#ifndef _SCENEOBJECT_H_
#define _SCENEOBJECT_H_

#include "openGL-headers.h"

#include "macros.h"
#include "objMesh.h"
#include "objMeshRender.h"
#include "vec3d.h"

class SceneObject
{
public:

  enum LightingModulationType {MODULATE, REPLACE};
  enum MipmapType {USEMIPMAP, NOMIPMAP};
  enum AnisotropicFilteringType {USEANISOTROPICFILTERING, NOANISOTROPICFILTERING};
  enum TextureTransparencyType {USETEXTURETRANSPARENCY, NOTEXTURETRANSPARENCY}; // enables 2-pass rendering for textures with transparencies

  // create a static scene object, by loading it from an Alias Wavefront OBJ file
  SceneObject(const char * filename);
  SceneObject(ObjMesh * objMesh, bool deepCopy = true);
  virtual ~SceneObject();

  // ==== render ====

  // if has displayList, calls displayList 
  // otherwise uses ObjMeshRender to render triangles with renderMode
  virtual void Render();
  virtual void RenderVertices();
  virtual void RenderEdges(); 
  virtual void RenderNormals();
  virtual void RenderFacesAndEdges();

  // only call glVertex3f to render the vertex position
  virtual void RenderVertex(int vertex);
  virtual void RenderVertices(int numVertices, int * vertexList); // 0-indexed vertices
  virtual void RenderVertices_Selection();

  virtual void RenderShadow(double ground[4], double light[4]);
  virtual void RenderGroup(const char * groupName);
  virtual void RenderGroup(int groupIndex);
  virtual void RenderEdgesInGroup(const char * groupName); // renders only the edges in the given group
  virtual void RenderEdgesInGroup(int groupIndex);

  // ==== display lists ====

  void BuildDisplayList();
  void PurgeDisplayList();

  // ==== mesh info and geometric queries ====

  inline int Getn() { return n; }
  inline int GetNumVertices() { return n; }
  inline int GetNumFaces() { return mesh->getNumFaces(); }
  inline const Vec3d & GetVertexPosition(int index) const { return mesh->getPosition(index); }
  inline ObjMesh * GetMesh() { return mesh; }
  inline ObjMeshRender * GetMeshRender() { return meshRender; }
  inline int GetRenderMode() { return renderMode; }

  // smallest ball radius that encloses the model, with the ball centered at the given centroid
  void ComputeMeshRadius(Vec3d & centroid, double * radius);
  // compute mesh centroid and smallest enclosing radius
  void ComputeMeshGeometricParameters(Vec3d * centroid, double * radius);
  // export mesh data
  void ExportMeshGeometry(int * numVertices, double ** vertices, int * numTriangles=NULL, int ** triangles=NULL);

  // finds the closest vertex using an exhaustive search
  // returns distance in "distance", if distance is not NULL
  // in this class, you can safely ignore the last parameter (keep it NULL)
  virtual int GetClosestVertex(Vec3d & queryPos, double * distance=NULL, double * auxVertexBuffer=NULL);

  // ==== texture mapping and materials ====

  // lightingModulation determines whether to multiply or replace the object color with the texture color
  // mipmap determines whether to use mipmapping for texture rendering
  // texturePool and updatePool should normally be set to NULL and 0.
  // These options exist so that you can share the same texture across multiple meshes.
  // If "texturePool" is not NULL, the loader will search for the texture in the pool of textures,
  // Only if not found, it will load the texture, otherwise, it will soft-link it.
  // If "updatePool" is 1, the loader will add a soft-link to any unique textures discovered in this object to the texturePool.
  // you must call this **after** OpenGL has been initialized!!!
  int SetUpTextures(LightingModulationType lightingModulation=MODULATE, MipmapType mipmap=USEMIPMAP,
      AnisotropicFilteringType anisotropicFiltering=USEANISOTROPICFILTERING, TextureTransparencyType=NOTEXTURETRANSPARENCY,
      std::vector<ObjMeshRender::Texture*> * texturePool=NULL, int updatePool=0);
  void EnableTextures();
  void DisableTextures();
  bool AreTexturesEnabled();
  inline bool HasTextures() { return hasTextures_; }

  void SetMaterialAlpha(double alpha); // sets the material alpha value for all the materials

  // ==== normals ====

  void BuildFaceNormals();
  void BuildNeighboringStructure();  // must be called before the vertex-normal functions below

  // second parameter is treshold angle for hard edges: 
  // if the normals of two nearby faces form an angle that's larger than thresholdAngle, it's considered as a hard edge
  void BuildVertexNormals(double thresholdAngle=85.0); // assumes pre-existing face normals
  void BuildNormals(double thresholdAngle=85.0); // builds both face and vertex normals
  void BuildNormalsFancy(double thresholdAngle=85.0);  // rebuilds facet normals + calls vertex-per-triangle normal update

  void SetNormalsToFaceNormals();

  // ==== vertex labeling ====

  // shows all point labels
  void ShowPointLabels();
  // shows point labels for points from k to l
  void ShowPointLabels(int k, int l);
  // renders vertex i in with given color and point size.
  // default: green color and with point size 8.0
  void HighlightVertex(int i, const Vec3d & color = Vec3d(0.0,1.0,0.0), double pointSize = 8.0) const;

  // ==== custom vetex color ====
  void EnableCustomColor();
  void DisableCustomColor();
  void SetCustomColor(const std::vector<Vec3d> & colors); // specific color for every mesh vertex

  // ==== render flat faces ====
  void EnableFlatFaces();
  void DisableFlatFaces();

  // model matrix for the shadow
  static void SetShadowingModelviewMatrix(double ground[4], double light[4]);

protected:
  void Construct();
  int n;
  unsigned int renderMode;
  TextureTransparencyType textureTransparency;

  bool deepCopy;
  ObjMesh * mesh;
  ObjMeshRender * meshRender;
  GLuint displayList;
  bool displayListExists;

  GLuint displayListEdges;
  bool displayListEdgesExists;

  void PrintBitmapString(float x,float y, float z, const char* s);
  void PrintBitmapInteger(float x,float y, float z, int i);

  bool hasTextures_;
};

#endif
