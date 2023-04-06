/*
  * Copyright (c) 2007, Carnegie Mellon University
  * All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions are met:
  *     * Redistributions of source code must retain the above copyright
  *       notice, this list of conditions and the following disclaimer.
  *     * Redistributions in binary form must reproduce the above copyright
  *       notice, this list of conditions and the following disclaimer in the
  *       documentation and/or other materials provided with the distribution.
  *     * Neither the name of Carnegie Mellon University, nor the
  *       names of its contributors may be used to endorse or promote products
  *       derived from this software without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND ANY
  * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY BE LIABLE FOR ANY
  * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <cassert>
#include "configFile.h"
#include "lighting.h"
#include "openGL-headers.h"


void Lighting::initialize()
{
  ambientIntensity = 0.0f;
  localViewer = true;
  twoSidedLighting = false;
  enableAmbientTerm = true;
  enableDiffuseTerm = true;
  enableSpecularTerm = true;
  memset(lightEnabled, 0, sizeof(lightEnabled));
  memset(lightPosition, 0, sizeof(lightPosition));
  for(int light = 0; light < 8; light++)
  {
    lightPosition[4*light+3] = 1.0f; //that's the direction component as in (x,y,z,d)
    lightIntensity[light] = 1.0f;
    lKa[light*4+0] = lKa[light*4+1] = lKa[light*4+2] = lightIntensity[light];
    lKd[light*4+0] = lKd[light*4+1] = lKd[light*4+2] = lightIntensity[light];
    lKs[light*4+0] = lKs[light*4+1] = lKs[light*4+2] = lightIntensity[light];
    lKa[light*4+3] = lKd[light*4+3] = lKs[light*4+3] = 1.0;
  }  
}

Lighting::Lighting()
{
  initialize();
}

void Lighting::buildConfig(ConfigFile & configFile)
{
  configFile.addOptionOptional("globalAmbientIntensity", &ambientIntensity, ambientIntensity);
  configFile.addOptionOptional("localViewer", &localViewer, localViewer);
  configFile.addOptionOptional("twoSidedLighting", &twoSidedLighting, twoSidedLighting);
  configFile.addOptionOptional("enableAmbientTerm", &enableAmbientTerm, enableAmbientTerm);
  configFile.addOptionOptional("enableDiffuseTerm", &enableDiffuseTerm, enableDiffuseTerm);
  configFile.addOptionOptional("enableSpecularTerm", &enableSpecularTerm, enableSpecularTerm);

  for(int light=0; light<8; light++)
  {
    char lightCh = '0' + light;
    char optionName[128];

    sprintf(optionName, "lightEnabled_%c", lightCh);
    configFile.addOptionOptional(optionName, &lightEnabled[light], lightEnabled[light]);

    sprintf(optionName, "position_%c_X", lightCh);
    configFile.addOptionOptional(optionName, &lightPosition[4*light+0], lightPosition[4*light+0]);
    sprintf(optionName, "position_%c_Y", lightCh);
    configFile.addOptionOptional(optionName, &lightPosition[4*light+1], lightPosition[4*light+1]);
    sprintf(optionName, "position_%c_Z", lightCh);
    configFile.addOptionOptional(optionName, &lightPosition[4*light+2], lightPosition[4*light+2]);
    sprintf(optionName, "position_%c_DIR", lightCh);
    configFile.addOptionOptional(optionName, &lightPosition[4*light+3], lightPosition[4*light+3]);

    sprintf(optionName, "lightIntensity_%c", lightCh);
    configFile.addOptionOptional(optionName, &lightIntensity[light], lightIntensity[light]);
  }
}

Lighting::Lighting(const char * configFilename)
{
  initialize();

  if (glGetError() != GL_NO_ERROR)
    printf("Warning: error in the OpenGL state at the beginning of lighting constructor.\n");

  ConfigFile configFile;

  buildConfig(configFile);

  if (configFile.parseOptions(configFilename) != 0)
    throw 1;


  //configFile.printOptions();

  #define TURNONLIGHT(i)\
   if (lightEnabled[i])\
     glEnable(GL_LIGHT##i);\
   else\
     glDisable(GL_LIGHT##i);\

  TURNONLIGHT(0);
  TURNONLIGHT(1);
  TURNONLIGHT(2);
  TURNONLIGHT(3);
  TURNONLIGHT(4);
  TURNONLIGHT(5);
  TURNONLIGHT(6);
  TURNONLIGHT(7);

  if (glGetError() != GL_NO_ERROR)
    printf("Warning: error in the OpenGL state in the Lighting constructor.\n");

  for(int light=0; light<8; light++)
  {
    if (enableAmbientTerm) 
    {
      lKa[light*4+0] = lKa[light*4+1] = lKa[light*4+2] = lightIntensity[light];
    }
    else
    {
      lKa[light*4+0] = lKa[light*4+1] = lKa[light*4+2] = 0.0f;
    }

    if (enableDiffuseTerm)
    {
      lKd[light*4+0] = lKd[light*4+1] = lKd[light*4+2] = lightIntensity[light];
    }
    else
    {
      lKd[light*4+0] = lKd[light*4+1] = lKd[light*4+2] = 0.0f;
    }

    if (enableSpecularTerm)
    {
      lKs[light*4+0] = lKs[light*4+1] = lKs[light*4+2] = lightIntensity[light];
    }
    else
    {
      lKs[light*4+0] = lKs[light*4+1] = lKs[light*4+2] = 0.0f;
    }

    // set alpha to 1.0
    lKa[light*4+3] = lKd[light*4+3] = lKs[light*4+3] = 1.0;
  }

  GLfloat aGa[] = { ambientIntensity, ambientIntensity, ambientIntensity, 1.0 };
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, aGa);
  glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, localViewer);
  glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, twoSidedLighting);
}

int Lighting::SaveConfig(const char * filename)
{
  ConfigFile configFile;

  buildConfig(configFile);

  return configFile.saveOptions(filename);
}

void Lighting::SetLightBox(const double lightBoxMin[3], const double lightBoxMax[3])
{
  float boxMin[3], boxMax[3];
  for(int i = 0; i < 3; i++)
  {
    boxMin[i] = lightBoxMin[i];
    boxMax[i] = lightBoxMax[i];
  }
  SetLightBox(boxMin, boxMax);
}

void Lighting::SetLightBox(const float lightBoxMin[3], const float lightBoxMax[3])
{
  GLfloat lP0[4] = { lightBoxMin[0], lightBoxMin[1], lightBoxMin[2], 1.0 };
  GLfloat lP1[4] = { lightBoxMax[0], lightBoxMin[1], lightBoxMin[2], 1.0 };
  GLfloat lP2[4] = { lightBoxMin[0], lightBoxMax[1], lightBoxMin[2], 1.0 };
  GLfloat lP3[4] = { lightBoxMax[0], lightBoxMax[1], lightBoxMin[2], 1.0 };
  GLfloat lP4[4] = { lightBoxMin[0], lightBoxMin[1], lightBoxMax[2], 1.0 };
  GLfloat lP5[4] = { lightBoxMax[0], lightBoxMin[1], lightBoxMax[2], 1.0 };
  GLfloat lP6[4] = { lightBoxMin[0], lightBoxMax[1], lightBoxMax[2], 1.0 };
  GLfloat lP7[4] = { lightBoxMax[0], lightBoxMax[1], lightBoxMax[2], 1.0 };
  GLfloat * lightPos[8] = { lP0, lP1, lP2, lP3, lP4, lP5, lP6, lP7 };
  for(int i = 0; i < 8; i++)
    memcpy(lightPosition + 4*i, lightPos[i], sizeof(float) * 4);
}

void Lighting::SetAllLightsEnabled(bool e)
{
  for(int i = 0; i < 8; i++)
  {
    lightEnabled[i] = e;
  }
  TURNONLIGHT(0);
  TURNONLIGHT(1);
  TURNONLIGHT(2);
  TURNONLIGHT(3);
  TURNONLIGHT(4);
  TURNONLIGHT(5);
  TURNONLIGHT(6);
  TURNONLIGHT(7);
}

void Lighting::SetAllLightsIntensity(float intensity)
{
  for(int i = 0; i < 8; i++)
    SetLightIntensity(i, intensity);
}

void Lighting::LightScene() const
{
  #define LIGHTSETUP(i)\
  if (lightEnabled[i])\
  {\
    glLightfv(GL_LIGHT##i, GL_POSITION, &lightPosition[4*i]);\
    glLightfv(GL_LIGHT##i, GL_AMBIENT, &lKa[4*i]);\
    glLightfv(GL_LIGHT##i, GL_DIFFUSE, &lKd[4*i]);\
    glLightfv(GL_LIGHT##i, GL_SPECULAR, &lKs[4*i]);\
  }

  LIGHTSETUP(0);
  LIGHTSETUP(1);
  LIGHTSETUP(2);
  LIGHTSETUP(3);
  LIGHTSETUP(4);
  LIGHTSETUP(5);
  LIGHTSETUP(6);
  LIGHTSETUP(7);

  glEnable(GL_LIGHTING);
}

void Lighting::SetLightEnabled(int lightID, bool e)
{
  if (lightEnabled[lightID] != e)
  {
    lightEnabled[lightID] = e;
    switch(lightID)
    {
      case 0: TURNONLIGHT(0); break;
      case 1: TURNONLIGHT(1); break;
      case 2: TURNONLIGHT(2); break;
      case 3: TURNONLIGHT(3); break;
      case 4: TURNONLIGHT(4); break;
      case 5: TURNONLIGHT(5); break;
      case 6: TURNONLIGHT(6); break;
      case 7: TURNONLIGHT(7); break;
    }
  }
}

void Lighting::SetLightPosition(int lightID, const float pos[4])
{
  memcpy(lightPosition + 4*lightID, pos, 4 * sizeof(float));
}

void Lighting::SetLightIntensity(int lightID, float intensity)
{
  lightIntensity[lightID] = intensity;
  if (enableAmbientTerm)
    lKa[lightID*4+0] = lKa[lightID*4+1] = lKa[lightID*4+2] = intensity;

  if (enableDiffuseTerm)
    lKd[lightID*4+0] = lKd[lightID*4+1] = lKd[lightID*4+2] = intensity;

  if (enableSpecularTerm)
    lKs[lightID*4+0] = lKs[lightID*4+1] = lKs[lightID*4+2] = intensity;

  //lKa[lightID*4+3] = lKd[lightID*4+3] = lKs[lightID*4+3] = 1.0;
}

void Lighting::SetAmbientEnabled(bool e)
{
  if (enableAmbientTerm != e)
  {
    enableAmbientTerm = e;
    for(int light = 0; light < 8; light++)
    {
      if (enableAmbientTerm) 
        lKa[light*4+0] = lKa[light*4+1] = lKa[light*4+2] = lightIntensity[light];
      else
        lKa[light*4+0] = lKa[light*4+1] = lKa[light*4+2] = 0.0f;
    }
  }
}

void Lighting::SetDiffuseEnabled(bool e)
{
  if (enableDiffuseTerm != e)
  {
    enableDiffuseTerm = e;
    for(int light = 0; light < 8; light++)
    {
      if (enableDiffuseTerm)
        lKd[light*4+0] = lKd[light*4+1] = lKd[light*4+2] = lightIntensity[light];
      else
        lKd[light*4+0] = lKd[light*4+1] = lKd[light*4+2] = 0.0f;
    }
  }
}

void Lighting::SetSpecularEnabled(bool e)
{
  if (enableSpecularTerm != e)
  {
    enableSpecularTerm = e;
    for(int light = 0; light < 8; light++)
    {
      if (enableSpecularTerm)
        lKs[light*4+0] = lKs[light*4+1] = lKs[light*4+2] = lightIntensity[light];
      else
        lKs[light*4+0] = lKs[light*4+1] = lKs[light*4+2] = 0.0f;
    }
  }
}

bool Lighting::ToggleLight(int lightID) 
{
  assert(lightID >= 0 && lightID < 8);
  SetLightEnabled(lightID, !lightEnabled[lightID]);
  return lightEnabled[lightID];
}

bool Lighting::HasLightEnabled() const
{
  for(int i = 0; i < 8; i++)
    if(lightEnabled[i] == true)
      return true;
  return false;
}

bool Lighting::HasLightDisabled() const
{
  for(int i = 0; i < 8; i++)
    if(lightEnabled[i] == false)
      return true;
  return false; 
}
