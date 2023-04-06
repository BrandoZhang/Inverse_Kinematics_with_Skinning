/*************************************************************************
 *                                                                       *
 * Vega FEM Simulation Library Version 4.0                               *
 *                                                                       *
 * "openGLHelper" library , Copyright (C) 2018 USC                       *
 * All rights reserved.                                                  *
 *                                                                       *
 * Code authors: Yijing Li, Jernej Barbic                                *
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

#ifndef HANDLERENDER_H
#define HANDLERENDER_H
#include "vec3d.h"
#include "mat3d.h"


// used to draw a handle with three arrows forming a coordinate system
class HandleRender
{
public:
  HandleRender();
  virtual ~HandleRender() {}

  // stencilValue: the stencil value used on the handle. Our code draws non-zero stencil values on rendered objects
  // so that when the user clicks on the object we know where she clicks.
  void render(const Vec3d & handlePosition, const Mat3d & rotation = Mat3d(1.0), int stencilValue = 1);

  void setLineWidth(double lw) { lineWidth = lw; }
  void setAxisLength(double al) { axisLength = al; }
  void setHandleReversed(bool r) { reverseDirection = r; }
  void setArrowHeadScale(double s) { arrowHeadScale = s; }

  // mouse location is used to highlight which axis the mouse is on
  // if mouseLocation is near the root of the axes, all three axes are highlighted
  void setMouseLocation(const Vec3d & mouseLocation); 
  // disable axis highlight
  void resetMouseLocation();

  // start handle dragging state
  // when dragging the handle, only the selected axis (arrow) is rendered according to the mosue location
  void dragHandle();
  // end handle dragging state
  // display all axes after the handle is released
  void releaseHandle(); 

  // whether the mouse is over a certain axis
  bool isMouseOverAxis(int axis) const { return (mouseOverMoveHandleAxis == 3 || mouseOverMoveHandleAxis == axis); }
  // whether the mouse is on the handle
  bool isMouseOnHandle() const { return mouseOverMoveHandleAxis != -1; }

protected:
  bool showAxis[3];
  double lineWidth;
  double axisLength;
  bool reverseDirection;
  double arrowHeadScale;

  Vec3d moveHandleAxesLC[3];
  Vec3d moveHandleAxesUC[3];

  // -1: not over any axis
  //  0: over the x-axis
  //  1: over the y-axis
  //  2: over the z-axis
  //  3: over all three axes
  int mouseOverMoveHandleAxis;
};


#endif
