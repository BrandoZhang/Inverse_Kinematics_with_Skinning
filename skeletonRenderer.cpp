#include "skeletonRenderer.h"
#include "openGLHelper.h"
#include "FK.h"
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

SkeletonRenderer::SkeletonRenderer(const FK * fk, double localAxisLength)
{
  renderedLocalAxisLength = localAxisLength;
  this->fk = fk;
}

void SkeletonRenderer::renderSkeleton() const
{
  // save attributes
  glPushAttrib(GL_LINE_BIT | GL_POINT_BIT | GL_CURRENT_BIT);

  // render joint position
  glColor3f(1,0,0);
  glPointSize(20);
  glBegin(GL_POINTS);
  for(int jointID = 0; jointID < fk->getNumJoints(); jointID++)
    Draw(fk->getJointGlobalPosition(jointID));
  glEnd();

  // render the connection between joints
  glLineWidth(5.0);
  glColor3f(1,1,0);
  glBegin(GL_LINES);
  for(int jointID = 0; jointID < fk->getNumJoints(); jointID++)
  {
    int parentID = fk->getJointParent(jointID);
    if (parentID < 0)
      continue;

    Draw(fk->getJointGlobalPosition(jointID), fk->getJointGlobalPosition(parentID));
  }
  glEnd();
  
  glPopAttrib();
}

void SkeletonRenderer::renderJointCoordAxes(int jointID) const
{
  // save attributes
  glPushAttrib(GL_LINE_BIT | GL_POINT_BIT | GL_CURRENT_BIT);
  Vec3d jointPos = fk->getJointGlobalPosition(jointID);
  // render coordinate frame of the joint
  glLineWidth(8.0);
  glBegin(GL_LINES);
  for(int d = 0; d < 3; d++)
  {
    Vec3d axisEndPosLocal(0.0);
    axisEndPosLocal[d] = renderedLocalAxisLength;
    Vec3d axisEndPosGlobal = fk->getJointGlobalTransform(jointID).transformPoint(axisEndPosLocal);
    Vec3d color(0,0,0);
    color[d] = 1.0;
    glColor3f(color[0], color[1] ,color[2]);
    Draw(jointPos);
    Draw(axisEndPosGlobal);
  }
  glEnd();

  glPopAttrib();
}

void SkeletonRenderer::renderJoint(int jointID) const
{
  // save attributes
  glPushAttrib(GL_LINE_BIT | GL_POINT_BIT | GL_CURRENT_BIT);

  // render joint position
  Vec3d jointPos = fk->getJointGlobalPosition(jointID);
  glColor3f(1,0,0);
  glPointSize(10);
  glBegin(GL_POINTS);
  Draw(jointPos);
  glEnd();

  // render descedents joints
  glLineWidth(5.0);
  glColor3f(1,0,0);
  vector<int> descedentIDs = fk->getJointDescendents(jointID);
  glBegin(GL_LINES);
  for(int childID : descedentIDs)
  {
  	int parentID = fk->getJointParent(childID);
    Draw(fk->getJointGlobalPosition(parentID));
    Draw(fk->getJointGlobalPosition(childID));
  }
  glEnd();
  glColor3f(0,0,1);
  glPointSize(10);
  glBegin(GL_POINTS);
  for(int childID : descedentIDs)
  {
    Draw(fk->getJointGlobalPosition(childID));
  }
  glEnd();
  glPopAttrib();
}

