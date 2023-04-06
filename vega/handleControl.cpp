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

#include "handleControl.h"
#include "openGLHelper.h"
#include <iostream>
using namespace std;

void HandleControl::setMouseButtonActivity(bool mouseDown, bool mouseOnObject, bool buildNewHandle, Vec3d worldPos, float zValue,
    function<int()> getClosestExistingHandle, function<pair<int,bool>()> addOrRemoveHandle)
{
  if (mouseDown && mouseOnObject == false) {
    selectedHandleVertex = -1;
    handleRender.resetMouseLocation();
    return;
  }

  if (mouseDown) {
    inDrag = true;
    if (buildNewHandle == false) {
      // we search the closest existing handle
      oldMouseWorldPos = worldPos;
      handleRender.dragHandle();
      IKZValue = zValue;
      if (handleRender.isMouseOnHandle() == false) // if the mouse is not on an existing handle
        selectedHandleVertex = getClosestExistingHandle();
      printf("Selected handle at: %d\n", selectedHandleVertex);
    }
    else { // build a new handle or delete an existing one
      auto p = addOrRemoveHandle();
      int clickedVertex = p.first;
      if (p.second) { // build a new handle
        selectedHandleVertex = clickedVertex;
      }
      else {
        if (selectedHandleVertex == clickedVertex)
        {
          selectedHandleVertex = -1;
          handleRender.resetMouseLocation();
        }
      }
    } // end buildNewHandle
  }
  else { // release button
    handleRender.releaseHandle();
    inDrag = false;
  }
}

void HandleControl::clearHandleSelection() {
  selectedHandleVertex = -1;
  inDrag = false;
  handleRender.resetMouseLocation();
}

void HandleControl::processHandleMovement(int x,int y, bool detailMotion, function<void(int vertex, Vec3d diff)> processDrag) {
  if (isInDrag()) {
    Vec3d worldPos(0.0);
    //      unprojectPointFromScreen(g_vMousePos[0], g_vMousePos[1], &worldPos[0]);
    unprojectPointFromScreen(x, y, IKZValue, &worldPos[0]);

    Vec3d posDiff = worldPos - oldMouseWorldPos;
    oldMouseWorldPos = worldPos;

    for(int i = 0; i < 3; i++)
      posDiff[i] = handleRender.isMouseOverAxis(i) ? posDiff[i] : (0.0);

    if (detailMotion)
      posDiff *= 0.1;

    if (len2(posDiff) > 0)
    {
//      cout << "draging..." << endl;
      // the user is dragging an IK vertex
      processDrag(selectedHandleVertex, posDiff);
    }
  }
}

void HandleControl::renderHandle(SphericalCamera * camera, const Vec3d & handlePosition, bool reverseHandle) {
  double lineWidth = 3.0*(16.0/camera->GetRadius());
  lineWidth = ((lineWidth > 4.0) ? 4.0 : (lineWidth < 3.0 ? 3.0 : lineWidth));
  double axisLength = 1.0/(16.0/camera->GetRadius());
  handleRender.setLineWidth(lineWidth);
  handleRender.setAxisLength(axisLength);
  handleRender.setHandleReversed(reverseHandle);

  double scale = 2.0/(16/camera->GetRadius());
  handleRender.setArrowHeadScale(scale);

  handleRender.render(handlePosition);
}
