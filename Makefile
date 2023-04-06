# CSCI 520 HW3 skinning and IK Makefile 
# Jernej Barbic, Yijing Li, USC

DRIVER_OBJECT_FILES = driver.o skinning.o FK.o IK.o skeletonRenderer.o
DRIVER_HEADERS = FK.h skinning.h IK.h minivectorTemplate.h skeletonRenderer.h
LIB_OBJECT_FILES = sceneObject.o sceneObjectWithRestPosition.o sceneObjectDeformable.o objMesh.o objMeshRender.o cameraLighting.o lighting.o vec3d.o listIO.o camera.o averagingBuffer.o inputDevice.o openGLHelper.o configFile.o mat4d.o mat3d.o handleControl.o handleRender.o matrixIO.o

CXX = g++
#CXXFLAGS= -g -std=c++11 -fsanitize=address -fsanitize=undefined
CXXFLAGS= -O3 -std=c++11 -DGL_SILENCE_DEPRECATION -Wno-deprecated-declarations -Wno-deprecated

ADOLC_ROOT=$(HOME)
#ADOLC_ROOT=$(HOME)/software
ADOLC_INCLUDE=-I$(ADOLC_ROOT)/adolc_base/include/
ADOLC_LIB=-ladolc -L$(ADOLC_ROOT)/adolc_base/lib64/

OPENGL_LIBS=-lGL -lGLU -lglut
#OPENGL_LIBS=-framework OpenGL -framework GLUT
#OPENGL_LIBS=-framework OpenGL /usr/local/Cellar/freeglut/3.0.0/lib/libglut.dylib

EIGEN_INCLUDE=-Ieigen/

INCLUDE = -Ivega/ $(ADOLC_INCLUDE) $(EIGEN_INCLUDE)

ALL = driver EigenSolveExample ADOLCExample
all: $(ALL)

driver: $(DRIVER_OBJECT_FILES) vega/libpartialVega.a
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ $(ADOLC_LIB) $(OPENGL_LIBS) -lm -o $@

vega/libpartialVega.a:  $(addprefix vega/, $(LIB_OBJECT_FILES))
	ar r $@ $^

$(DRIVER_OBJECT_FILES): %.o: %.cpp $(DRIVER_HEADERS) vega/*.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

$(LIB_OBJECT_FILES): %.o: %.cpp vega/*.h
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $^ -o $@

EigenSolveExample: EigenSolveExample.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ -o $@

ADOLCExample: ADOLCExample.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ $(ADOLC_LIB) -o $@

clean:
	-rm -rf core *.o vega/*.o vega/libpartialVega.a $(ALL)

