#include "mat4d.h"

const Mat4d Mat4d::Identity(1.0);
const Mat4d Mat4d::Zero(0.0);

static_assert(sizeof(Mat4d) == sizeof(double) * 16, "Size of Mat4d should be sizeof(double) * 16");