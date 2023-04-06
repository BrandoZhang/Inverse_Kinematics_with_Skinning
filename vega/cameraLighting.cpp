/*************************************************************************
 *                                                                       *
 * Vega FEM Simulation Library Version 4.0                               *
 *                                                                       *
 * "lighting" library , Copyright (C) 2018 USC                           *
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

#include "cameraLighting.h"
#include "configFile.h"
#include "openGL-headers.h"
#include "camera.h"
#include <iostream>
using namespace std;

void CameraLighting::initialize()
{
  if (lightEnabled)
    glEnable(GL_LIGHT0);
  else
    glDisable(GL_LIGHT0);

  glDisable(GL_LIGHT1);
  glDisable(GL_LIGHT2);
  glDisable(GL_LIGHT3);
  glDisable(GL_LIGHT4);
  glDisable(GL_LIGHT5);
  glDisable(GL_LIGHT6);
  glDisable(GL_LIGHT7);

  if (glGetError() != GL_NO_ERROR)
    printf("Warning: error in the OpenGL state in the cameraLighting constructor.\n");

  GLfloat aGa[] = { ambientIntensity, ambientIntensity, ambientIntensity, 1.0 };
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, aGa);
  glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, localViewer);
  glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, twoSidedLighting);
}

CameraLighting::CameraLighting()
{
  initialize();
}

CameraLighting::CameraLighting(const char * configFilename)
{
  if (glGetError() != GL_NO_ERROR)
    printf("Warning: error in the OpenGL state at the beginning of camera lighting constructor.\n");

  ConfigFile configFile;
  configFile.addOptionOptional("globalAmbientIntensity", &ambientIntensity, ambientIntensity);
  configFile.addOptionOptional("localViewer", &localViewer, localViewer);
  configFile.addOptionOptional("twoSidedLighting", &twoSidedLighting, twoSidedLighting);
  configFile.addOptionOptional("enableAmbientTerm", &enableAmbientTerm, enableAmbientTerm);
  configFile.addOptionOptional("enableDiffuseTerm", &enableDiffuseTerm, enableDiffuseTerm);
  configFile.addOptionOptional("enableSpecularTerm", &enableSpecularTerm, enableSpecularTerm);
  configFile.addOptionOptional("lightEnabled", &lightEnabled, lightEnabled);
  configFile.addOptionOptional("lightIntensity", &lightIntensity, lightIntensity);

  if (configFile.parseOptions(configFilename) != 0)
    throw 1;

  initialize();
}

void CameraLighting::LightScene(const SphericalCamera * camera)
{
  double cx[3], cy[3], cz[3];
  camera->Get3DAxes(cx, cy, cz);
  for(int i = 0; i < 3; i++)
    lightPosition[i] = cz[i];
  lightPosition[3] = 0.0;

  glLightfv(GL_LIGHT0, GL_POSITION, &lightPosition[0]);

  if (enableAmbientTerm)
    lKa[0] = lKa[1] = lKa[2] = lightIntensity;
  else
    lKa[0] = lKa[1] = lKa[2] = 0;

  if (enableAmbientTerm)
    lKd[0] = lKd[1] = lKd[2] = lightIntensity;
  else
    lKd[0] = lKd[1] = lKd[2] = 0;

  if (enableAmbientTerm)
    lKs[0] = lKs[1] = lKs[2] = lightIntensity;
  else
    lKs[0] = lKs[1] = lKs[2] = 0;

  lKa[3]= lKd[3] = lKs[3] = 1.0;

  glLightfv(GL_LIGHT0, GL_AMBIENT, &lKa[0]);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, &lKd[0]);
  glLightfv(GL_LIGHT0, GL_SPECULAR, &lKs[0]);

//  cout << "light scene" << endl;
//  cout << lightPosition[0] << " " << lightPosition[1] << " " << lightPosition[2] << endl;
//  cout << "lka " << lKa[0] << " " << lKd[0] << " " << lKs[0] << " " << lightEnabled << endl;

  if (lightEnabled)
    glEnable(GL_LIGHT0);
  else
    glDisable(GL_LIGHT0);

  glEnable(GL_LIGHTING);
}

int CameraLighting::SaveConfig(const char * filename)
{
  ConfigFile configFile;
  configFile.addOptionOptional("globalAmbientIntensity", &ambientIntensity, ambientIntensity);
  configFile.addOptionOptional("localViewer", &localViewer, localViewer);
  configFile.addOptionOptional("twoSidedLighting", &twoSidedLighting, twoSidedLighting);
  configFile.addOptionOptional("enableAmbientTerm", &enableAmbientTerm, enableAmbientTerm);
  configFile.addOptionOptional("enableDiffuseTerm", &enableDiffuseTerm, enableDiffuseTerm);
  configFile.addOptionOptional("enableSpecularTerm", &enableSpecularTerm, enableSpecularTerm);
  configFile.addOptionOptional("lightEnabled", &lightEnabled, lightEnabled);
  configFile.addOptionOptional("lightIntensity", &lightIntensity, lightIntensity);

  return configFile.saveOptions(filename);
}

