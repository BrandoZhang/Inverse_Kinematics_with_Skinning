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

#include "handleRender.h"
#include "openGLHelper.h"


//static int numArrowHeadVertices = 22;
static double arrowHeadVertices[66]={0.057018,0.000000,-0.018526,0.048503,0.000000,-0.035239,0.035239,0.000000,-0.048503,0.018526,0.000000,-0.057018,0.000000,0.000000,-0.059953,-0.018526,0.000000,-0.057018,-0.035239,0.000000,-0.048503,-0.048503,0.000000,-0.035239,-0.057018,0.000000,-0.018526,-0.059953,0.000000,0.000000,-0.057018,0.000000,0.018526,-0.048503,0.000000,0.035239,-0.035239,0.000000,0.048503,-0.018526,0.000000,0.057018,-0.000000,0.000000,0.059953,0.018526,0.000000,0.057018,0.035239,0.000000,0.048503,0.048503,0.000000,0.035239,0.057018,0.000000,0.018526,0.059953,0.000000,0.000000,0.000000,0.281334,0.000000,-0.000000,0.000000,-0.000000};
//static double arrowHeadFaceVertexNormals[147]={0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.930171, 0.208421, -0.302230, 0.791250, 0.208421, -0.574877, -0.000000, 1.000000, 0.000001, 0.574877, 0.208421, -0.791250, 0.302231, 0.208421, -0.930171, -0.000000, 0.208421, -0.978039, -0.302231, 0.208421, -0.930171, -0.574877, 0.208421, -0.791250, -0.791250, 0.208421, -0.574877, -0.930171, 0.208421, -0.302231, -0.978039, 0.208421, 0.000000, -0.930171, 0.208421, 0.302231, -0.791250, 0.208421, 0.574877, -0.574877, 0.208421, 0.791250, -0.302231, 0.208421, 0.930171, 0.000000, 0.208421, 0.978039, 0.302231, 0.208421, 0.930171, 0.574877, 0.208421, 0.791250, 0.791250, 0.208421, 0.574877, 0.930171, 0.208421, 0.302231, 0.978039, 0.208421, 0.000001, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, -1.000000, 0.000000};
static int arrowHeadNumFaces = 40;
static int arrowHeadFaceIndices[120]={17, 22, 18,18, 22, 19,19, 22, 20,20, 22, 1,1, 22, 2,1, 2, 21,2, 3, 21,3, 4, 21,4, 5, 21,5, 6, 21,6, 7, 21,7, 8, 21,8, 9, 21,9, 10, 21,10, 11, 21,11, 12, 21,12, 13, 21,13, 14, 21,14, 15, 21,15, 16, 21,16, 17, 21,17, 18, 21,18, 19, 21,19, 20, 21,20, 1, 21,17, 16, 22,16, 15, 22,15, 14, 22,14, 13, 22,22, 13, 12,7, 22, 8,12, 11, 22,11, 10, 22,10, 9, 22,22, 9, 8,2, 22, 3,7, 6, 22,6, 5, 22,5, 4, 22,22, 4, 3};
//static int arrowHeadFaceVertexNormalIndices[120]={1,2,3,3,2,4,4,2,5,5,2,6,6,2,7,8,9,10,9,11,10,11,12,10,12,13,10,13,14,10,14,15,10,15,16,10,16,17,10,17,18,10,18,19,10,19,20,10,20,21,10,21,22,10,22,23,10,23,24,10,24,25,10,25,26,10,26,27,10,27,28,10,28,8,10,29,30,31,30,32,31,32,33,31,33,34,31,31,34,35,36,37,38,39,40,37,40,41,37,41,42,37,37,42,38,43,44,45,46,47,44,47,48,44,48,49,44,44,49,45};
// one 3x3 rotation matrix per each axis
static Mat3d arrowHeadRotationMatrices[3] = { Mat3d(0,1,0, -1,0,0,  0,0,1),
                                              Mat3d(1,0,0,  0,1,0,  0,0,1),
                                              Mat3d(-1,0,0, 0,0,-1, 0,1,0) };


HandleRender::HandleRender(): lineWidth(3.0), axisLength(1.0), reverseDirection(false), arrowHeadScale(1.3), mouseOverMoveHandleAxis(-1)
{
  for(int i = 0; i < 3; i++)
  {
    moveHandleAxesLC[i] = moveHandleAxesUC[i] = Vec3d(0.);
    showAxis[i] = true;
  }
}

void HandleRender::resetMouseLocation()
{
  mouseOverMoveHandleAxis = -1;
  showAxis[0] = showAxis[1] = showAxis[2] = true;
}

void HandleRender::dragHandle()
{
  if (mouseOverMoveHandleAxis != -1) // If the mouse cursor is over the handle
  {
    // it's only one axis selected, or all three axes selected
    showAxis[0] = showAxis[1] = showAxis[2] = false;
    if (mouseOverMoveHandleAxis == 0)
      showAxis[0] = true;
    else if (mouseOverMoveHandleAxis == 1)
      showAxis[1] = true;
    else if (mouseOverMoveHandleAxis == 2)
      showAxis[2] = true;
    else if (mouseOverMoveHandleAxis == 3)
      showAxis[0] = showAxis[1] = showAxis[2] = true;
  }
}

void HandleRender::releaseHandle()
{
  showAxis[0] = showAxis[1] = showAxis[2] = true;
}

void HandleRender::setMouseLocation(const Vec3d & mouseLocation)
{
  mouseOverMoveHandleAxis = -1;
  bool overAxis[3];
  overAxis[0] = overAxis[1] = overAxis[2] = false;
  for(int axisIndex=0; axisIndex<3; axisIndex++)
  {
    //moveHandleAxesLC[axisIndex].print();// printf("\n");
    //moveHandleAxesUC[axisIndex].print();// printf("\n");
    overAxis[axisIndex] =
           ((moveHandleAxesLC[axisIndex][0] < mouseLocation[0]) && (mouseLocation[0] < moveHandleAxesUC[axisIndex][0])
        && (moveHandleAxesLC[axisIndex][1] < mouseLocation[1]) && (mouseLocation[1] < moveHandleAxesUC[axisIndex][1])
        && (moveHandleAxesLC[axisIndex][2] < mouseLocation[2]) && (mouseLocation[2] < moveHandleAxesUC[axisIndex][2]));
  }
  if ((overAxis[0] && overAxis[1]) || (overAxis[0] && overAxis[2]) || (overAxis[1] && overAxis[2]))
    mouseOverMoveHandleAxis = 3;
  else if (overAxis[0])
    mouseOverMoveHandleAxis = 0;
  else if (overAxis[1])
    mouseOverMoveHandleAxis = 1;
  else if (overAxis[2])
    mouseOverMoveHandleAxis = 2;

//  if (mouseOverMoveHandleAxis > -1)
//    cout << "axis " << mouseOverMoveHandleAxis << endl;
}

void HandleRender::render(const Vec3d & handlePosition, const Mat3d & rotation, int stencilValue)
{
  float previousLineWidth;
  glGetFloatv(GL_LINE_WIDTH, &previousLineWidth);
  //glEnable(GL_LIGHTING);
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_LIGHTING);

  //float lineAmbient[4] = {0.1, 0.1, 0.1, 1.0};
  //float lineDiffuse[4];
  //lineDiffuse[3] = 1.0;
  //float lineSpecular[4] = {1.0, 1.0, 1.0, 1.0};
  //float lineShininess = 1.0;
  //glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, lineAmbient);
  //glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, lineSpecular);
  //glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, lineShininess);

  Vec3d axisOffset[3];
  Vec3d axisEndPoint[3];

  glLineWidth(lineWidth);
  // draw the handle axes
  glBegin(GL_LINES);
  for(int i = 0; i < 3; i++)
  {
    if (!showAxis[i])
      continue;
    //lineDiffuse[0] = lineDiffuse[1] = lineDiffuse[2] = 0.0;
    //lineDiffuse[i] = 1.0;
    //glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, lineDiffuse);
    double vertexColor[3] = {0, 0, 0};
    vertexColor[i] = 1.0;
    glColor3f(vertexColor[0], vertexColor[1], vertexColor[2]);
    glVertex3f(handlePosition[0], handlePosition[1], handlePosition[2]);

    axisOffset[i] = Vec3d(0.);
    axisOffset[i][i] = axisLength;
    axisOffset[i] = rotation * axisOffset[i];

    vertexColor[i] = 0.5;
    glColor3f(vertexColor[0], vertexColor[1], vertexColor[2]);
    axisEndPoint[i] = handlePosition + (reverseDirection ? -1.0 : 1.0) * axisOffset[i];
    glVertex3f(axisEndPoint[i][0], axisEndPoint[i][1], axisEndPoint[i][2]);
  }
  glEnd();

  // enhance the axis selectable area by hacking the stencil buffer
  int previousStencilFunc = GL_ALWAYS;
  int previousStencilValue = 0;
  glGetIntegerv(GL_STENCIL_FUNC, &previousStencilFunc);
  glGetIntegerv(GL_STENCIL_REF, &previousStencilValue);
  glStencilFunc(GL_ALWAYS, stencilValue, ~(0u)); //always pass stencil test, stencil renference value is stencilValue (default 1)
  glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
  glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
  glLineWidth(lineWidth*200.0);
  glBegin(GL_LINES);
  for(int i = 0; i < 3; i++)
  {
    if (!showAxis[i])
      continue;
    glVertex3f(handlePosition[0], handlePosition[1], handlePosition[2]);
    glVertex3f(axisEndPoint[i][0], axisEndPoint[i][1], axisEndPoint[i][2]);
  }
  glEnd();
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

  glLineWidth(previousLineWidth);

  // draw arrow head
  glBegin(GL_TRIANGLES);
  for(int axisIndex=0; axisIndex<3; axisIndex++)
  {
    if (!showAxis[axisIndex])
      continue;

    moveHandleAxesLC[axisIndex] = handlePosition - Vec3d(0.1 * arrowHeadScale);
    moveHandleAxesUC[axisIndex] = handlePosition + Vec3d(0.1 * arrowHeadScale);

    double scale = arrowHeadScale;
    if ((mouseOverMoveHandleAxis == axisIndex) || (mouseOverMoveHandleAxis == 3))
      scale *= 1.3;

    for(int i=0; i<arrowHeadNumFaces; i++)
    {

      for(int v=0; v<3; v++)
      {
        int vIndex = arrowHeadFaceIndices[3*i+v] - 1;

        double vertexColor[3] = {0, 0, 0};
        vertexColor[axisIndex] = 1.0;
        if (vIndex == 20)
          vertexColor[axisIndex] = 0.0;
        glColor3f(vertexColor[0], vertexColor[1], vertexColor[2]);

        Mat3d arrowHeadR = rotation * arrowHeadRotationMatrices[axisIndex];

        Vec3d vertexPosition = handlePosition + (reverseDirection ? -1.0 : 1.0) * (axisOffset[axisIndex] +
            scale * (arrowHeadR * Vec3d(&arrowHeadVertices[3*vIndex])) );

        if (vertexPosition[0] > moveHandleAxesUC[axisIndex][0])
          moveHandleAxesUC[axisIndex][0] = vertexPosition[0];
        if (vertexPosition[1] > moveHandleAxesUC[axisIndex][1])
          moveHandleAxesUC[axisIndex][1] = vertexPosition[1];
        if (vertexPosition[2] > moveHandleAxesUC[axisIndex][2])
          moveHandleAxesUC[axisIndex][2] = vertexPosition[2];

        if (vertexPosition[0] < moveHandleAxesLC[axisIndex][0])
          moveHandleAxesLC[axisIndex][0] = vertexPosition[0];
        if (vertexPosition[1] < moveHandleAxesLC[axisIndex][1])
          moveHandleAxesLC[axisIndex][1] = vertexPosition[1];
        if (vertexPosition[2] < moveHandleAxesLC[axisIndex][2])
          moveHandleAxesLC[axisIndex][2] = vertexPosition[2];

        //glNormal3f(vVertexNormal[0], vVertexNormal[1], vVertexNormal[2]);
        glVertex3f(vertexPosition[0], vertexPosition[1], vertexPosition[2]);
      }
    }
  }
  glEnd();

  glStencilFunc(previousStencilFunc, previousStencilValue, ~(0u));
}
