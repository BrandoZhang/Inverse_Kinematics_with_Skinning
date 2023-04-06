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

#include "openGLHelper.h"
#include "macros.h"
#include <iostream>
using namespace std;
//static void * fonts[] =
//{
//  GLUT_BITMAP_9_BY_15,
//  GLUT_BITMAP_TIMES_ROMAN_10,
//  GLUT_BITMAP_TIMES_ROMAN_24
//};

void UnitCube()
{
  glBegin(GL_TRIANGLES);

  glNormal3f(0,-1,0);  

  glVertex3f(0,0,0); // front
  glVertex3f(1,0,0);
  glVertex3f(1,0,1);

  glVertex3f(0,0,0);
  glVertex3f(1,0,1);
  glVertex3f(0,0,1);

  glNormal3f(0,1,0);  

  glVertex3f(0,1,0); // back
  glVertex3f(1,1,1);
  glVertex3f(1,1,0);

  glVertex3f(0,1,1);
  glVertex3f(1,1,1);
  glVertex3f(0,1,0);

  glNormal3f(1,0,0);  

  glVertex3f(1,0,0); // right
  glVertex3f(1,1,0);
  glVertex3f(1,1,1);

  glVertex3f(1,0,0);
  glVertex3f(1,1,1);
  glVertex3f(1,0,1);

  glNormal3f(-1,0,0);  

  glVertex3f(0,0,0); // left
  glVertex3f(0,1,1);
  glVertex3f(0,1,0);

  glVertex3f(0,0,1);
  glVertex3f(0,1,1);
  glVertex3f(0,0,0);

  glNormal3f(0,0,1);  

  glVertex3f(0,0,1); // top
  glVertex3f(1,0,1);
  glVertex3f(1,1,1);

  glVertex3f(0,0,1);
  glVertex3f(1,1,1);
  glVertex3f(0,1,1);

  glNormal3f(0,0,-1);  

  glVertex3f(0,0,0); // bottom
  glVertex3f(1,1,0);
  glVertex3f(1,0,0);

  glVertex3f(0,1,0);
  glVertex3f(1,1,0);
  glVertex3f(0,0,0);

  glEnd();
}

void UnitCubeWireframe()
{
  glBegin(GL_LINES); 
    glVertex3f(0,0,0);
    glVertex3f(1,0,0);
    glVertex3f(0,1,0);
    glVertex3f(1,1,0);
    glVertex3f(0,0,0);
    glVertex3f(0,1,0);
    glVertex3f(1,0,0);
    glVertex3f(1,1,0);

    glVertex3f(0,0,1);
    glVertex3f(1,0,1);
    glVertex3f(0,1,1);
    glVertex3f(1,1,1);
    glVertex3f(0,0,1);
    glVertex3f(0,1,1);
    glVertex3f(1,0,1);
    glVertex3f(1,1,1);

    glVertex3f(0,0,0);
    glVertex3f(0,0,1);

    glVertex3f(0,1,0);
    glVertex3f(0,1,1);

    glVertex3f(1,0,0);
    glVertex3f(1,0,1);

    glVertex3f(1,1,0);
    glVertex3f(1,1,1);
  glEnd();
}

void RenderWireframeBox(double bmin[3], double bmax[3])
{
  glBegin(GL_LINES); 
    glVertex3f(bmin[0],bmin[1],bmin[2]);
    glVertex3f(bmax[0],bmin[1],bmin[2]);
    glVertex3f(bmin[0],bmax[1],bmin[2]);
    glVertex3f(bmax[0],bmax[1],bmin[2]);
    glVertex3f(bmin[0],bmin[1],bmin[2]);
    glVertex3f(bmin[0],bmax[1],bmin[2]);
    glVertex3f(bmax[0],bmin[1],bmin[2]);
    glVertex3f(bmax[0],bmax[1],bmin[2]);

    glVertex3f(bmin[0],bmin[1],bmax[2]);
    glVertex3f(bmax[0],bmin[1],bmax[2]);
    glVertex3f(bmin[0],bmax[1],bmax[2]);
    glVertex3f(bmax[0],bmax[1],bmax[2]);
    glVertex3f(bmin[0],bmin[1],bmax[2]);
    glVertex3f(bmin[0],bmax[1],bmax[2]);
    glVertex3f(bmax[0],bmin[1],bmax[2]);
    glVertex3f(bmax[0],bmax[1],bmax[2]);

    glVertex3f(bmin[0],bmin[1],bmin[2]);
    glVertex3f(bmin[0],bmin[1],bmax[2]);

    glVertex3f(bmin[0],bmax[1],bmin[2]);
    glVertex3f(bmin[0],bmax[1],bmax[2]);

    glVertex3f(bmax[0],bmin[1],bmin[2]);
    glVertex3f(bmax[0],bmin[1],bmax[2]);

    glVertex3f(bmax[0],bmax[1],bmin[2]);
    glVertex3f(bmax[0],bmax[1],bmax[2]);
  glEnd();

}

void RenderSphere(float x, float y, float z)
{
  glPushMatrix();
  glTranslatef(x,y,z);
  glutSolidSphere(0.02,20,20);
  glPopMatrix();
}

void ChangeStateToRenderingText(int w, int h)
{
  glPushAttrib   ( GL_ENABLE_BIT | GL_VIEWPORT_BIT | GL_TRANSFORM_BIT | GL_LIGHTING_BIT ) ;
  glDisable      ( GL_LIGHTING   ) ;
  glDisable      ( GL_FOG        ) ;
  glDisable      ( GL_TEXTURE_2D ) ;
  glDisable      ( GL_DEPTH_TEST ) ;
  glDisable      ( GL_CULL_FACE  ) ;
  glEnable       ( GL_ALPHA_TEST ) ;
  glEnable       ( GL_BLEND ) ;
  glAlphaFunc    ( GL_GREATER, 0.1f ) ;
  glBlendFunc    ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ) ;

  glMatrixMode   ( GL_PROJECTION ) ;
  glPushMatrix   () ;
  glLoadIdentity () ;
  gluOrtho2D     ( 0, w, 0, h ) ;
  glMatrixMode   ( GL_MODELVIEW ) ;
  glPushMatrix   () ;
  glLoadIdentity () ;
}

void RestoreStateFromeRenderingText()
{
  glMatrixMode   ( GL_PROJECTION ) ;
  glPopMatrix    () ;
  glMatrixMode   ( GL_MODELVIEW ) ;
  glPopMatrix    () ;
  glPopAttrib    () ;
}

void OutputText(float X, float Y, const char *string, void * font)
{
  glRasterPos2f(X, Y);
  for(size_t i = 0, len = strlen(string); i < len; i++) 
    glutBitmapCharacter(font, string[i]);
}

// void OutputText(float relx, float rely, int windowWidth, int windowHeight, const char * string, void * font)
// {
//   if(relx < 0)
//     relx = windowWidth + relx;
//   if(rely < 0)
//     rely = windowHeight + rely;
//   OutputText(-1 + 2.0 * relx / windowWidth, -1 + 2.0 * rely / windowHeight, string, font);
// }

void OutputStrokeText(float X, float Y, float fontSizeScale, const char * string, void * font)
{
  glPushMatrix();
  glTranslatef(X, Y, -1);
  glScalef(0.001 * fontSizeScale, 0.001 * fontSizeScale, 1);
  for(size_t i = 0, len = strlen(string); i < len; i++) 
    glutStrokeCharacter(font, string[i]);
  glPopMatrix();
}

// void OutputStrokeText(float relx, float rely, int windowWidth, int windowHeight, float fontSizeScale, const char * string, void * font)
// {
//   if(relx < 0)
//     relx = windowWidth + relx;
//   if(rely < 0)
//     rely = windowHeight + rely;
//   OutputStrokeText(-1 + 2.0 * relx / windowWidth, -1 + 2.0 * rely / windowHeight, fontSizeScale, string, font);
// }

void PrintGLerror( const char *msg )
{
 GLenum errCode;
 const GLubyte *errStr;

 if ((errCode = glGetError()) != GL_NO_ERROR) 
 {
    errStr = gluErrorString(errCode);
    printf("OpenGL ERROR: %s: %s\n", errStr, msg);
	exit(1);
 }
}



/*
void RenderTexturesIntoDisplayList()

{
  glPushMatrix();
  //glTranslatef(xLow, yLow, zLow);
  //glScalef(xSize, ySize, zSize);

  // draw the walls
  glEnable(GL_TEXTURE_2D);
  
  glBindTexture(GL_TEXTURE_2D,texName3);

  glBegin(GL_QUADS); // left wall
  glTexCoord2f(0,0); glVertex3f(0 ,0 , 0 );
  glTexCoord2f(1,0); glVertex3f(0 ,1 , 0 );
  glTexCoord2f(1,1); glVertex3f(0 ,1 , 1 );
  glTexCoord2f(0,1); glVertex3f(0 ,0 , 1 );
  glEnd();

  glBegin(GL_QUADS); // far wall
  glTexCoord2f(0,0); glVertex3f(0 ,1 ,0 );
  glTexCoord2f(1,0); glVertex3f(1 ,1 ,0 );
  glTexCoord2f(1,1); glVertex3f(1 ,1 ,1 );
  glTexCoord2f(0,1); glVertex3f(0 ,1 ,1 );
  glEnd();
  
  glBegin(GL_QUADS); // right wall
  glTexCoord2f(0,0); glVertex3f(1 ,1 , 0 );
  glTexCoord2f(1,0); glVertex3f(1 ,0 , 0 );
  glTexCoord2f(1,1); glVertex3f(1 ,0 , 1 );
  glTexCoord2f(0,1); glVertex3f(1 ,1 , 1 );
  glEnd();

  glBegin(GL_QUADS); // near wall
  glTexCoord2f(0,0); glVertex3f(1 ,0 ,0 );
  glTexCoord2f(1,0); glVertex3f(0 ,0 ,0 );
  glTexCoord2f(1,1); glVertex3f(0 ,0 ,1 );
  glTexCoord2f(0,1); glVertex3f(1 ,0 ,1 );
  glEnd();

  glBindTexture(GL_TEXTURE_2D,texName2);
  
  glBegin(GL_QUADS); // floor
  glTexCoord2f(0,0); glVertex3f(0 ,0 ,0 );
  glTexCoord2f(1,0); glVertex3f(1 ,0 ,0 );
  glTexCoord2f(1,1); glVertex3f(1 ,1 ,0 );
  glTexCoord2f(0,1); glVertex3f(0 ,1 ,0 );
  glEnd();

  glBindTexture(GL_TEXTURE_2D,texName1);

  glBegin(GL_QUADS); // ceiling
  glTexCoord2f(0,0); glVertex3f(0 ,1 ,1 );
  glTexCoord2f(1,0); glVertex3f(1 ,1 ,1 );
  glTexCoord2f(1,1); glVertex3f(1 ,0 ,1 );
  glTexCoord2f(0,1); glVertex3f(0 ,0 ,1 );

  glEnd();


  glDisable(GL_TEXTURE_2D);

  glPopMatrix();

}
*/


void DetermineCameraParameters(double centerX, double centerY, double centerZ, double modelRadius,
							   double * focusX, double * focusY, double * focusZ,
							   double * cameraRadius, double * zNear, double * zFar)
{
  *focusX = centerX;
  *focusY = centerY;
  *focusZ = centerZ;

  *cameraRadius = modelRadius * 3;
  
  *zNear = *cameraRadius * 0.01;
  *zFar = *cameraRadius * 100;
}


void DrawArrow( float px, float py, float pz,
    float nx, float ny, float nz,
    double arrowEndWidth, double arrowEndLength )
{
  GLdouble normal[3], cross[3], zaxis[3];
  GLdouble angle;

  #define DegreesToRadians     (3.14159265 / (GLfloat) 180.0)

  normal[0] = nx; normal[1] = ny; normal[2] = nz;
  double len = sqrt(nx*nx + ny*ny + nz*nz);

  if (len < 1E-6)
    return;


  glPushMatrix();
    glTranslatef( px, py, pz );
    glBegin( GL_LINES );
      glVertex3f( 0.0, 0.0, 0.0 );
      glVertex3f( nx, ny, nz );
    glEnd();

    // normalize the normal vector 
    normal[0] /= len;
    normal[1] /= len;
    normal[2] /= len;

    // determine angle between z axis and normal 
    zaxis[0] = 0; zaxis[1] = 0; zaxis[2] = 1;
    angle = acos( zaxis[0]*normal[0] + zaxis[1]*normal[1] + zaxis[2]*normal[2] )/DegreesToRadians;

    if ( angle != 0.0 ) 
    {
      // find the axis of rotation 
      CROSSPRODUCT( zaxis[0], zaxis[1], zaxis[2], 
          normal[0], normal[1], normal[2],
          cross[0], cross[1], cross[2] );
      glRotatef( angle, cross[0], cross[1], cross[2] );
    }

    // move to end of normal vector 
    glTranslatef( 0.0, 0.0, len );
	/*
	glScalef(0.3,0.3,0.3);
    #ifdef EIFFEL140VOX
  	  glScalef(2.0,2.0,2.0);
    #endif
    #ifdef SPOON100VOX
  	  glScalef(2.0,2.0,2.0);
    #endif
	*/
    glutSolidCone( len*arrowEndWidth, len*arrowEndLength, 24, 3 );
  glPopMatrix();
}

void DrawArrow(const Vec3d & p, const Vec3d & n,
    double arrowEndWidth, double arrowEndLength)
{
  DrawArrow(p[0], p[1], p[2], n[0], n[1], n[2], arrowEndWidth, arrowEndLength);
}

void RenderLine(const Vec3d & v0, const Vec3d & v1)
{
  glBegin(GL_LINES);
  Draw(v0, v1);
  glEnd();
}

void JetColorMap(double x, double color[3])
{
  double a; // alpha

  if (x < 0)
  {
    color[0] = 0;
    color[1] = 0;
    color[2] = 0;
    return;
  } 
  else if (x < 0.125) 
  {
    a = x / 0.125;
    color[0] = 0;
    color[1] = 0;
    color[2] = 0.5 + 0.5 * a;
    return;
  }
  else if (x < 0.375) 
  {
    a = (x - 0.125) / 0.25;
    color[0] = 0;
    color[1] = a;
    color[2] = 1;
    return;
  }
  else if (x < 0.625) 
  {
    a = (x - 0.375) / 0.25;
    color[0] = a;
    color[1] = 1;
    color[2] = 1 - a;
    return;
  }
  else if (x < 0.875) 
  {
    a = (x - 0.625) / 0.25;
    color[0] = 1;
    color[1] = 1 - a;
    color[2] = 0;
    return;
  }
  else if (x <= 1.0) 
  {
    a = (x - 0.875) / 0.125;
    color[0] = 1 - 0.5 * a;
    color[1] = 0;
    color[2] = 0;
    return;
  }
  else 
  {
    color[0] = 1;
    color[1] = 1;
    color[2] = 1;
    return;
  }
}

void TransparentSphere(GLuint solidSphereList, GLuint wireSphereList, double x, double y, double z, double radius)
{
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  glPushMatrix();
  glTranslatef(x,y,z);
  glColor3f(1,0,0);
  glBegin(GL_POINTS);
    glVertex3f(0,0,0);
  glEnd();
  //glColor4f(1,1,1,0.35);
  glScalef(radius,radius,radius);
  glCallList(solidSphereList);
  //glutSolidSphere(radius,25,25);
  glDisable(GL_BLEND);
  //glColor3f(0,0,0);
  glCallList(wireSphereList);
  //glutWireSphere(radius,25,25);
  glPopMatrix();
}

void BuildSphereDisplayList(GLuint * solidSphereList, GLuint * wireSphereList)
{ 
  *solidSphereList = glGenLists(1);
  glNewList(*solidSphereList, GL_COMPILE);
    glColor4f(1,1,1,0.35);
    glutSolidSphere(1.0,25,25);
  glEndList();

  *wireSphereList = glGenLists(1);
  glNewList(*wireSphereList, GL_COMPILE);
    glColor3f(0,0,0);
    glutWireSphere(1.0,25,25);
  glEndList();
} 

char * DuplicateString(const char * s)
{
  // strdup sometimes causes problems, so we resort to this
  char * p = (char*) malloc (sizeof(char) * (strlen(s) + 1));
  memcpy(p, s, sizeof(char) * (strlen(s) + 1));
  return p;
}


void RenderAxes(double axisLength) 
{
  GLboolean lightingEnabled = glIsEnabled(GL_LIGHTING);
  GLfloat color[4];
  glGetFloatv(GL_CURRENT_COLOR, color);
  glDisable(GL_LIGHTING);
//  glEnable(GL_COLOR_MATERIAL);
  glBegin(GL_LINES);
  for (int i = 0; i<3; i++) 
  {
    float color[3] = { 0, 0, 0 };
    color[i] = 1.0;
    glColor3fv(color);
    float vertex[3] = {0, 0, 0};
    vertex[i] = axisLength;
    glVertex3fv(vertex);
    glVertex3f(0, 0, 0);
  }
  glEnd();
  if (lightingEnabled)
    glEnable(GL_LIGHTING);
  glColor3fv(color);
}

void RenderLocator(const Vec3d & center, double r)
{
  GLboolean lightingEnabled = glIsEnabled(GL_LIGHTING);
  GLfloat color[4];
  glGetFloatv(GL_CURRENT_COLOR, color);
  glDisable(GL_LIGHTING);

  glBegin(GL_LINES);
  glColor3f(1, 0.5, 0.5);
  Draw(center + Vec3d(r, 0, 0), center - Vec3d(r, 0, 0));
  glColor3f(0.5, 1, 0.5);
  Draw(center + Vec3d(0, r, 0), center - Vec3d(0, r, 0));
  glColor3f(0.5, 0.5, 1);
  Draw(center + Vec3d(0, 0, r), center - Vec3d(0, 0, r));
  glEnd();

  if (lightingEnabled)
    glEnable(GL_LIGHTING);
  glColor3fv(color);
}

float readPixelDepthValue(int x, int y)
{
  GLint view[4];
  glGetIntegerv (GL_VIEWPORT, view);

  int winX = x;
  int winY = view[1]+view[3]-1-y;

  float zValue = 0.0f;
  glReadPixels(winX,winY,1,1, GL_DEPTH_COMPONENT, GL_FLOAT, &zValue);
  return zValue;
}

void unprojectPointFromScreen(int x, int y, double worldPosition[3], GLubyte * stencilValue, float * depthValue)
{
  GLdouble model[16], proj[16];
  glGetDoublev (GL_MODELVIEW_MATRIX, model);
  glGetDoublev (GL_PROJECTION_MATRIX, proj);

  GLint view[4];
  glGetIntegerv (GL_VIEWPORT, view);

  int winX = x;
  int winY = view[1]+view[3]-1-y;

  float zValue = 0;
  glReadPixels(winX,winY,1,1, GL_DEPTH_COMPONENT, GL_FLOAT, &zValue);
  if (depthValue)
    *depthValue = zValue;

  if (stencilValue)
  {
    *stencilValue = 0;
    glReadPixels(winX, winY, 1, 1, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, stencilValue);
  }

  GLdouble worldX, worldY, worldZ;
  gluUnProject(winX, winY, zValue, model, proj, view, &worldX, &worldY, &worldZ);
  worldPosition[0] = worldX;
  worldPosition[1] = worldY;
  worldPosition[2] = worldZ;
}

void unprojectPointFromScreen(int x, int y, double depthValue, double worldPosition[3], GLubyte * stencilValue)
{
  GLdouble model[16], proj[16];
  glGetDoublev (GL_MODELVIEW_MATRIX, model);
  glGetDoublev (GL_PROJECTION_MATRIX, proj);

  GLint view[4];
  glGetIntegerv (GL_VIEWPORT, view);

  int winX = x;
  int winY = view[1]+view[3]-1-y;

  if (stencilValue)
  {
    *stencilValue = 0;
    glReadPixels(winX, winY, 1, 1, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, stencilValue);
  }

  GLdouble worldX, worldY, worldZ;
  gluUnProject(winX, winY, depthValue, model, proj, view, &worldX, &worldY, &worldZ);
  worldPosition[0] = worldX;
  worldPosition[1] = worldY;
  worldPosition[2] = worldZ;
}

PolygonOffsetFillState::PolygonOffsetFillState(GLfloat inputFactor, GLfloat inputUnits, OffsetType type)
{
  glGetFloatv(GL_POLYGON_OFFSET_FACTOR, &prevFactor);
  glGetFloatv(GL_POLYGON_OFFSET_UNITS, &prevUnits);
  
  if (type == OT_ABSOLUTE) { glPolygonOffset(inputFactor, inputUnits); }
  else { glPolygonOffset(prevFactor + inputFactor, prevUnits + inputUnits); } // relative
  prevOffsetEnabled = glIsEnabled(GL_POLYGON_OFFSET_FILL);
  glEnable(GL_POLYGON_OFFSET_FILL);
}

void PolygonOffsetFillState::restore()
{
  if (restored == true) return;
  restored = true;
  if (prevOffsetEnabled == false)
    glDisable(GL_POLYGON_OFFSET_FILL);
  glPolygonOffset(prevFactor, prevUnits);
}

void PolygonOffsetFillState::changeOffset(GLfloat inputFactor, GLfloat inputUnits, OffsetType type)
{
  if (type == OT_ABSOLUTE) { glPolygonOffset(inputFactor, inputUnits); }
  else { glPolygonOffset(prevFactor + inputFactor, prevUnits + inputUnits); } // relative
}
