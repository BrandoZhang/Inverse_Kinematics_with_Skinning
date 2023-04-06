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

#include "inputDevice.h"
#include "openGLHelper.h"

InputDevice::InputDevice()
{
  memset(mousePos, 0, sizeof(mousePos));
  leftMouseButton = middleMouseButton = rightMouseButton = false;
  shift = alt = ctrl = false;
}

void InputDevice::setButton(int button, int state)
{
  switch (button)
  {
    case GLUT_LEFT_BUTTON:
      leftMouseButton = (state==GLUT_DOWN);
    break;

    case GLUT_MIDDLE_BUTTON:
      middleMouseButton = (state==GLUT_DOWN);
    break;

    case GLUT_RIGHT_BUTTON:
      rightMouseButton = (state==GLUT_DOWN);
    break;
  }

  setModifiers();
}

void InputDevice::setModifiers()
{
  shift = (glutGetModifiers() == GLUT_ACTIVE_SHIFT);
  alt = (glutGetModifiers() == GLUT_ACTIVE_ALT);
  ctrl = (glutGetModifiers() == GLUT_ACTIVE_CTRL);
}
