/*************************************************************************
 *                                                                       *
 * Vega FEM Simulation Library Version 4.0                               *
 *                                                                       *
 * "animationHelper" library , Copyright (C) 2018 USC                    *
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

#ifndef HANDLECONTROL_H
#define HANDLECONTROL_H

#include "handleRender.h"
#include "vec3d.h"
#include "camera.h"

#include <functional>

class HandleControl
{
public:
  // called when button activity happened
  // mouseDown: true, mouse pressed; false, mouse released
  // updateHandle: true, call addOrRemoveHandle() to create/remove handles
  //               false, call getClosestExistingHandle() to select a handle
  void setMouseButtonActivity(bool mouseDown, bool mouseOnObject, bool updateHandle, Vec3d worldPos, float zValue,
      std::function<int()> getClosestExistingHandle, std::function<std::pair<int,bool>()> addOrRemoveHandle);

  // respond to handle movement by calling processDrag()
  void processHandleMovement(int x,int y, bool detailMotion, std::function<void(int vertex, Vec3d diff)> processDrag);

  void setMousePosition(const Vec3d & worldPos) { handleRender.setMouseLocation(worldPos); }

  int getSelectedHandle() const { return selectedHandleVertex; }

  bool isHandleSelected() const { return selectedHandleVertex >= 0; }

  bool isInDrag() const { return inDrag && selectedHandleVertex >= 0; }

  void clearHandleSelection();

  void renderHandle(SphericalCamera * camera, const Vec3d & handlePosition, bool reverseHandle);

protected:
  int selectedHandleVertex = -1;
  HandleRender handleRender;
  Vec3d oldMouseWorldPos{0.0};
  bool inDrag = false;
  float IKZValue = 0.0f;
};


#endif
