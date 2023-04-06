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

#ifndef CAMERALIGHTING_H
#define CAMERALIGHTING_H

#include <stdlib.h>
#include <string.h>

class SphericalCamera;

class CameraLighting
{
public:
  CameraLighting();
  CameraLighting(const char * configFilename);

  // call this inside your OpenGL display routine, after setting up the modelview and projection matrices
  void LightScene(const SphericalCamera * camera);

  // pos is a 4-float array representing generalized coord. of the light source
  inline void GetLightPosition(float pos[4]) const;
  inline void GetLightAmbientTerm(float Ka[4]) const;
  inline void GetLightDiffuseTerm(float Kd[4]) const;
  inline void GetLightSpecularTerm(float Ks[4]) const;
  inline float GetLightIntensity() const;
  inline float GetAmbientIntensity() const;
  inline bool IsLightEnabled() const;
  inline bool IsAmbientEnabled() const { return enableAmbientTerm; }
  inline bool IsDiffuseEnabled() const { return enableDiffuseTerm; }
  inline bool IsSpecularEnabled() const { return enableSpecularTerm; }

  inline void SetLightEnabled(bool e) { lightEnabled = e; }
  inline void SetLightIntensity(float intensity) { lightIntensity = intensity; }
  inline void SetAmbientEnabled(bool e) { enableAmbientTerm = e; }
  inline void SetDiffuseEnabled(bool e) { enableDiffuseTerm = e; }
  inline void SetSpecularEnabled(bool e) { enableSpecularTerm = e; }

  // toggle light, return whether the toggled light is enabled
  inline bool ToggleLight() { lightEnabled = !lightEnabled; return lightEnabled; }

  // save lighting config to file
  int SaveConfig(const char * filename);

protected:
  void initialize();

  float ambientIntensity = 0.0f;
  bool localViewer = true, twoSidedLighting = false;
  bool lightEnabled = true;
  float lightPosition[4] = {0.0};
  float lKa[4] = {0.0}, lKd[4] = {0.0}, lKs[4] = {0.0};
  float lightIntensity = 1.0;
  bool enableAmbientTerm = true, enableDiffuseTerm = true, enableSpecularTerm = true;
};

inline float CameraLighting::GetLightIntensity() const
{
  return lightIntensity;
}

inline void CameraLighting::GetLightPosition(float pos[4]) const
{
  memcpy(pos, &(lightPosition[0]), 4 * sizeof(float));
}

inline void CameraLighting::GetLightAmbientTerm(float Ka[4]) const
{
  memcpy(Ka, &(lKa[0]), 4* sizeof(float));
}

inline void CameraLighting::GetLightDiffuseTerm(float Kd[4]) const
{
  memcpy(Kd, &(lKd[0]), 4* sizeof(float));
}

inline void CameraLighting::GetLightSpecularTerm(float Ks[4]) const
{
  memcpy(Ks, &(lKs[0]), 4* sizeof(float));
}

inline float CameraLighting::GetAmbientIntensity() const
{
  return ambientIntensity;
}

inline bool CameraLighting::IsLightEnabled() const
{
  return lightEnabled;
}
#endif
