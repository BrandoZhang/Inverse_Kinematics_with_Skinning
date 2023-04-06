/*************************************************************************
 *                                                                       *
 * Vega FEM Simulation Library Version 4.0                               *
 *                                                                       *
 * "openGLHelper" library , Copyright (C) 2007 CMU, 2009 MIT, 2018 USC   *
 * All rights reserved.                                                  *
 *                                                                       *
 * Code author: Jernej Barbic                                            *
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
  Assorted OpenGL "helper" routines.
*/

#ifndef _OPENGLHELPER_H_
#define _OPENGLHELPER_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <assert.h>
#include <cassert>
#include "openGL-headers.h"
#include "vec3d.h"

// ============================= Rendering Texts =========================


// this function saves some current openGL states and set up for rendering texts on 
// window space: left: x=0, right: x = windowWidth, bottom: y = 0, top: y = windowHeight
void ChangeStateToRenderingText(int windowWidth, int windowHeight);

// restore states saved from ChangeStateToRenderingText()
void RestoreStateFromeRenderingText();

// helper class to set and restore states for text rendering
class TextState
{
public:
  TextState(int windowWidth, int windowHeight) { ChangeStateToRenderingText(windowWidth, windowHeight); }
  ~TextState() { if (restored == false) RestoreStateFromeRenderingText(); }

  void restore() { if (restored == false) { RestoreStateFromeRenderingText(); restored = true; } }
protected:
  bool restored = false;
};

// render text on screen using glutBitmapCharacter
// call ChangeStateToRenderingText() first to set up. Alternatively, OpenGL modelview 
// matrix need to be identy and projection matrix should be gluOrtho2D(0, w, 0, h) to 
// show the text correctly
// font size of bitmap characters cannot be controlled
// color can be controlled by calling glColor3f beforehand                         ^ y=windowHeight
// font can be one of                                                              |
//  GLUT_BITMAP_9_BY_15,                                                           |
//  GLUT_BITMAP_TIMES_ROMAN_10,                                      x=0  -------------------> x=windowWidth
//  GLUT_BITMAP_TIMES_ROMAN_24,                                                    | 
//  etc.                                                                           |
// (X, Y): window space coordinates                                                | y=0
void OutputText(float X, float Y, const char * string, void * font = GLUT_BITMAP_9_BY_15);
// (relx, rely): shifted screen space coordinates in pixel units; origin is at the lower-left corner of the screen
// input args windowWidth, windowHeight are needed to convert pixel units into original screen space coords.
// if relx >= 0, relx starts at left edge of the window; otherwise, relx_p starts at right edge of the window
// if rely >= 0, rely starts at bottom edge of the window; otherwise, rely_p starts at top edge of the window
// void OutputTextRelative(float relx, float rely, int windowWidth, int windowHeight, const char * string, void * font = GLUT_BITMAP_9_BY_15);

// same requirement as OutputText()
// render text using stroke characters
// font can be one of
// GLUT_STROKE_ROMAN, and
// GLUT_STROKE_MONO_ROMAN.
// fontSizeScale: stroke characters can be scaled and their original sizes are huge compared to Bitmap characters. 
// fontSizeScale=1 gives characters about three times larger than bitmap characters
void OutputStrokeText(float X, float Y, float fontSizeScale, const char * string, void * font = GLUT_STROKE_ROMAN);
// other parameters same as OutputText(float relx_p, float rely_p, int windowWidth, int windowHeight, const char * string, void * font)
// void OutputStrokeText(float relx_p, float rely_p, int windowWidth, int windowHeight, float fontSizeScale, const char * string, void * font = GLUT_STROKE_ROMAN);

// ============================= Rendering Shapes =========================

// call glBegin()... glEnd() to render a unit cube / cube wireframe. No color specified
void UnitCube();
void UnitCubeWireframe();

void RenderWireframeBox(double bmin[3], double bmax[3]);

// draw an arrow starting from point p and end at point (p+n)
// arrowEndWidth: relative base size of the arrow head
// arrowEndLength: relative height of the arrow head
void DrawArrow(float px, float py, float pz,
    float nx, float ny, float nz,
    double arrowEndWidth, double arrowEndLength);
void DrawArrow(const Vec3d & p, const Vec3d & n,
    double arrowEndWidth, double arrowEndLength);

inline void Draw(const Vec3d & p) { glVertex3f(p[0], p[1], p[2]); }
inline void Draw(const Vec3d & p0, const Vec3d & p1) { Draw(p0); Draw(p1); }
inline void Draw(const Vec3d & p0, const Vec3d & p1, const Vec3d & p2) { Draw(p0); Draw(p1); Draw(p2); }

inline void DrawNormal(const Vec3d & p) { glNormal3f(p[0], p[1], p[2]); }

// use glBegin(GL_LINES) to draw a line segment
void RenderLine(const Vec3d & v0, const Vec3d & v1);


void DetermineCameraParameters(double centerX, double centerY, double centerZ, double modelRadius, double * focusX, double * focusY, double * focusZ, double * cameraRadius, double * zNear, double * zFar);

// convert scalar [0,1] to RGB color
// values <0 or >1 are clamped
void JetColorMap(double x, double color[3]);

void RenderSphere(float x, float y, float z);
void BuildSphereDisplayList(GLuint * solidSphereList, GLuint * wireSphereList);
void TransparentSphere(GLuint solidSphereList, GLuint wireSphereList, double x, double y, double z, double radius);

char * DuplicateString(const char * s);

void PrintGLerror( const char *msg );

void RenderAxes(double axisLength);

// render a locator at center made with 3 line segments, each with different color and same length of 2.0 * radius
void RenderLocator(const Vec3d & center, double radius = 1.0);

// given (x,y) from GLUT mouseButtonActivity callback, get unprojected world position and optionally stencil value and depth value
void unprojectPointFromScreen(int x, int y, double worldPosition[3], GLubyte * stencilValue = NULL, float * depthValue = NULL);

float readPixelDepthValue(int x, int y);
// given (x,y)  from GLUT mouseButtonActivity callback and depth value from readPixelDepthValue(x,y), 
// get unprojected world position and optionally stencil value
void unprojectPointFromScreen(int x, int y, double depthValue, double worldPosition[3], GLubyte * stencilValue = NULL);

// manage OpenGL polygon offset fill state
class PolygonOffsetFillState
{
public:
  enum OffsetType
  {
    OT_RELATIVE,
    OT_ABSOLUTE
  };
  PolygonOffsetFillState(GLfloat inputFactor, GLfloat inputUnits, OffsetType type = OT_RELATIVE);

  ~PolygonOffsetFillState() { restore(); }

  void changeOffset(GLfloat inputFactor, GLfloat inputUnits, OffsetType type = OT_RELATIVE);

  void restore();
protected:
  bool restored = false;
  GLfloat prevFactor = 0.0f;
  GLfloat prevUnits = 0.0f;
  bool prevOffsetEnabled = false;
};

#endif

