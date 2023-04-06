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

#ifndef INPUTDEVICE_H
#define INPUTDEVICE_H 

// GLUT keyboard & mouse control
class InputDevice
{ 
public:
  InputDevice();
  virtual ~InputDevice() {}

  // called in mouseButtonActivity(int button, int state, int x, int y), mouseDrag(int x, int y) and mouseNoDrag(int x, int y), set mousePos
  void setMousePos(int x, int y) { mousePos[0] = x; mousePos[1] = y; }

  // called in mouseButtonActivity(int button, int state, int x, int y), set mouse button and shift/alt/ctrl
  void setButton(int button, int state);

  // called in SpecialKeysFunc, set shift/alt/ctrl
  void setModifiers();

  // get keyboard & mouse control
  int getMousePosX() const { return mousePos[0]; }
  int getMousePosY() const { return mousePos[1]; }
  bool leftMouseButtonDown() const { return leftMouseButton; }
  bool middleMouseButtonDown() const { return middleMouseButton; }
  bool rightMouseButtonDown() const { return rightMouseButton; }
  bool shiftPressed() const { return shift; }
  bool altPressed() const { return alt; }
  bool ctrlPressed() const { return ctrl; }

protected:
  int mousePos[2];
  bool leftMouseButton, middleMouseButton, rightMouseButton;
  bool shift, alt, ctrl;
};


#endif
