#ifndef SKELETONRENDERER_H
#define SKELETONRENDERER_H

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

class FK;

// A class to render a skeleton using OpenGL.

class SkeletonRenderer
{
public:
 
  // localAxisLength: Length of the displayed coordinate axes at each joint, in scene units.
  SkeletonRenderer(const FK * fk, double localAxisLength);

  void renderJoint(int jointID) const;
  void renderSkeleton() const;
  void renderJointCoordAxes(int jointID) const;

private:
  const FK * fk = nullptr;
  double renderedLocalAxisLength = 1.0;
};

#endif

