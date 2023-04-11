// A driver to perform inverse kinametics with skinning.

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li


#include <vector>
#include <set>
#include <stdlib.h>
#include <stdio.h>
#include <climits>
#include <iostream>
#include <math.h>
#include <time.h>
#include <cmath>

#ifndef ASIO_STANDALONE
#define ASIO_STANDALONE
#endif // !ASIO_STANDALONE
#ifndef _WEBSOCKETPP_CPP11_TYPE_TRAITS_
#define _WEBSOCKETPP_CPP11_TYPE_TRAITS_
#endif // !_WEBSOCKETPP_CPP11_TYPE_TRAITS_
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <json.hpp>

#include "sceneObjectDeformable.h"
#include "lighting.h"
#include "cameraLighting.h"
#include "openGL-headers.h"
#include "camera.h"
#include "objMesh.h"
#include "performanceCounter.h"
#include "averagingBuffer.h"
#include "inputDevice.h"
#include "openGLHelper.h"
#include "valueIndex.h"
#include "configFile.h"
#include "skinning.h"
#include "FK.h"
#include "IK.h"
#include "handleControl.h"
#include "skeletonRenderer.h"
#ifdef WIN32
#include <windows.h>
#endif

#include <adolc/adolc.h>
using namespace std;

using json = nlohmann::json;

typedef websocketpp::server<websocketpp::config::asio> server;

static string meshFilename;
static string configFilename;
static string screenshotBaseName;
static string jointHierarchyFilename;
static string jointWeightsFilename;
static string jointRestTransformsFilename;

static bool fullScreen = 0;
static bool showAxes = false;
static bool showWireframe = true;
static bool showObject = true;
static bool useLighting = true;
static double allLightsIntensity = 1.0;

static Vec3d modelCenter(0.0);
static double modelRadius = 1.0;
static ObjMesh * mesh = nullptr;
static SceneObjectDeformable * meshDeformable = nullptr;

static FK * fk = nullptr;
static IK * ik = nullptr;
static Skinning * skinning = nullptr;
static SkeletonRenderer * skeletonRenderer = nullptr;

static bool renderSkeleton = true;
static int curJointID = -1;

static SphericalCamera * camera = nullptr;
static int windowWidth = 800, windowHeight = 600;
static double zNear = 0.001, zFar = 1000;
static int selectedVertex = -1;

static int windowID = 0;
static int graphicsFrameID = 0;

static Lighting * lighting = nullptr;
static CameraLighting * cameraLighting = nullptr;

static InputDevice id;
static bool reverseHandle = false;
static HandleControl handleControl;

static PerformanceCounter counter, titleBarCounter;
static int titleBarFrameCounter = 0;
static AveragingBuffer fpsBuffer(5);

static vector<int> IKJointIDs;
static vector<Vec3d> IKJointPos;

IKMethod IKMETHOD = Tikhonov;
SkinningMethod SKINNINGMETHOD = LinearBlendSkinning;

const int mediaPipeHand2ProjectJointID[21] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 15, 16, 17, 18};
const int IKJointID2MediaPipeHand[5] = {4, 8, 12, 16, 20};
std::queue<std::vector<Vec3d>> end_effector_positions;
std::vector<Vec3d> last_used_end_effector_positions;
std::mutex end_effector_positions_mutex;
std::condition_variable end_effector_positions_cv;

//======================= Functions =============================

// Handle incoming landmarks from websockets
void on_message(server* s, websocketpp::connection_hdl hdl, server::message_ptr msg) {
    try {
        // Parse the received JSON data
        json data = json::parse(msg->get_payload());

        // Create a vector to store the end effector positions for this prediction
        std::vector<Vec3d> current_prediction_positions;

        // Extract landmarks from the JSON data
        for (int i = 0; i < 21; i++) {
            auto& lm = data["landmarks"][i];
            Vec3d position(lm["x"], lm["y"], lm["z"]);

            // Add the end effector position to the current prediction positions
            current_prediction_positions.push_back(position);
        }

        // Push the new end effector positions into the queue
        {
            std::lock_guard<std::mutex> lock(end_effector_positions_mutex);
            end_effector_positions.push(current_prediction_positions);
        }
        end_effector_positions_cv.notify_one();
    }
    catch (std::exception& e) {
        std::cout << "Error parsing JSON: " << e.what() << std::endl;
    }
}

static void updateSkinnedMesh()
{
  vector<Vec3d> newPos(meshDeformable->GetNumVertices());
  double * newPosv = (double*)newPos.data();

  fk->computeJointTransforms();

  skinning->applySkinning(fk->getJointSkinTransforms(), newPosv);
  for(size_t i = 0; i < mesh->getNumVertices(); i++)
    mesh->setPosition(i, newPos[i]);

  meshDeformable->BuildNormals();
}

static void resetSkinningToRest()
{
  fk->resetToRestPose();
  updateSkinnedMesh();
  for(size_t i = 0; i < IKJointIDs.size(); i++)
  {
    IKJointPos[i] = fk->getJointGlobalPosition(IKJointIDs[i]);
  }
  handleControl.clearHandleSelection();
  curJointID = -1;

  cout << "reset mesh to rest" << endl;
}

static void idleFunction()
{
  glutSetWindow(windowID);
  counter.StopCounter();
  // double dt = counter.GetElapsedTime();
  counter.StartCounter();

  // Take appropriate action in case the user is dragging a vertex.
  auto processDrag = [&](int vertex, Vec3d posDiff)
  {
    if (len2(posDiff) > 0 && handleControl.isHandleSelected())
    {
      IKJointPos[handleControl.getSelectedHandle()] += posDiff;
    }
  };
  handleControl.processHandleMovement(id.getMousePosX(), id.getMousePosY(), id.shiftPressed(), processDrag);

  if (fk->getNumJoints() == 23)  // hand model
  {
      // Check for new end effector positions
      std::unique_lock<std::mutex> lock(end_effector_positions_mutex);
      end_effector_positions_cv.wait_for(lock, std::chrono::milliseconds(5), [] { return !end_effector_positions.empty(); });

      while (!end_effector_positions.empty()) {
          // Initialize the last used end effector positions
          if (last_used_end_effector_positions.empty()) 
          {
			  last_used_end_effector_positions = end_effector_positions.front();
			  end_effector_positions.pop();
			  continue;
		  }
          // Get the next end effector position from the queue
          std::vector<Vec3d> positions = end_effector_positions.front();
          end_effector_positions.pop();
          lock.unlock();

          // Set the actual end effector position
          for (int idx = 0; idx < 5; idx++)
          {
              int jointIDinMediaPipeHand = IKJointID2MediaPipeHand[idx];
              Vec3d jointPos = fk->getJointGlobalPosition(IKJointIDs[idx]);
              Vec3d offset = positions[jointIDinMediaPipeHand] - last_used_end_effector_positions[jointIDinMediaPipeHand];
              IKJointPos[idx] += 8 * offset;
          }
          last_used_end_effector_positions = positions;

          // Lock the mutex again for the next iteration
          lock.lock();
      }
  }

  const int maxIKIters = 10;
  const double maxOneStepDistance = modelRadius / 1000;

  ik->doIK(IKJointPos.data(), fk->getJointEulerAngles());

  updateSkinnedMesh();

  titleBarFrameCounter++;
  // update title bar at 4 Hz
  titleBarCounter.StopCounter();
  double elapsedTime = titleBarCounter.GetElapsedTime();
  if (elapsedTime >= 1.0 / 4)
  {
    titleBarCounter.StartCounter();
    double fps = titleBarFrameCounter / elapsedTime;
    fpsBuffer.addValue(fps);

    // update menu bar
    char windowTitle[4096];
    const char * IKMethodName;
    if (IKMETHOD == PseudoInverse)
      IKMethodName = "PseudoInverse";
    else if (IKMETHOD == Tikhonov)
      IKMethodName = "Tikhonov";
    else
      IKMethodName = "Kernel Trick Tikhonov";
    const char * SkinningMethodName;
    if (SKINNINGMETHOD == DualQuaternionSkinning)
      SkinningMethodName = "Dual Quaternion Skinning";
    else
      SkinningMethodName = "Linear Blend Skinning";

    sprintf(windowTitle, "Vertices: %d | %.1f FPS | graphicsFrame %d | IK Method: %s | Skinning Method: %s", meshDeformable->Getn(), fpsBuffer.getAverage(), graphicsFrameID, IKMethodName, SkinningMethodName);
    glutSetWindowTitle(windowTitle);
    titleBarFrameCounter = 0;
  }
  graphicsFrameID++;
  glutPostRedisplay();
}

static void reshape(int x, int y)
{
  glViewport(0,0,x,y);
  glMatrixMode(GL_PROJECTION); 
  glLoadIdentity(); 
  windowWidth = x;
  windowHeight = y;

  // Calculate the aspect ratio of the window
  gluPerspective(45.0f, 1.0 * windowWidth / windowHeight, zNear, zFar);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

static void displayFunction()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  camera->Look(); // calls gluLookAt

  glDisable(GL_LIGHTING);

  glLineWidth(1.0);
  if (showAxes)
    RenderAxes(1);

  if (useLighting)
  {
    glEnable(GL_LIGHTING);
    if (cameraLighting)
      cameraLighting->LightScene(camera);
    else if (lighting)
      lighting->LightScene();
  }
  else
    glDisable(GL_LIGHTING);

  glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE); //only when stencil pass and z-buffer pass, set stencil value to stencil reference
  glStencilFunc(GL_ALWAYS, 1, ~(0u));        //always pass stencil test, stencil renference value is 1

  if(true)
  {
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

//    glEnable(GL_POLYGON_OFFSET_FILL);
//    glPolygonOffset(1.0, 1.0);
//    glDrawBuffer(GL_NONE);

    /***********************************
     *    render transparent object
     ***********************************/
//    glDisable(GL_POLYGON_OFFSET_FILL);
//    glDrawBuffer(GL_BACK);
    glDisable(GL_BLEND);
  }

  glColor3f(0.9,0.9,0.9);
  if (showObject)
  {
    meshDeformable->Render();
  }
  glColor3f(0,0,0);
  if (showWireframe)
    meshDeformable->RenderEdges();

  glDisable(GL_BLEND);
  glDisable(GL_LIGHTING);
  glPointSize(10.0); 

  // ---------------------------------------------------------
  //    now rendering non-material/non-texture stuff here
  if (selectedVertex >= 0)
  {
    glColor3f(1,0,0);
    glBegin(GL_POINTS);
    Draw(mesh->getPosition(selectedVertex));
    glEnd();
  }

  glDisable(GL_DEPTH_TEST);
  if (renderSkeleton)
  {
    skeletonRenderer->renderSkeleton();
  }
  if (curJointID >= 0 && curJointID < fk->getNumJoints())
  {
    skeletonRenderer->renderJoint(curJointID);
  }
  for(int jointID : IKJointIDs)
  {
    skeletonRenderer->renderJointCoordAxes(jointID);
  }
  glEnable(GL_DEPTH_TEST);

  glStencilFunc(GL_ALWAYS, 0, ~(0u)); // always pass stencil test, stencil renference value is set to 0
  // render the vertex currently being manipulated via IK
  if (handleControl.isHandleSelected())
  {
    int handleID = handleControl.getSelectedHandle();
    Vec3d handlePos = fk->getJointGlobalPosition(IKJointIDs[handleID]);
    glColor3f(1,0,0);
    glPointSize(8.0);
    Draw(handlePos);

    // render the moving handle at location IKJointPos[handleID]
    handleControl.renderHandle(camera, IKJointPos[handleID], reverseHandle);
  }

  glutSwapBuffers();
}

static void keyboardFunc(unsigned char key, int x, int y)
{
  switch (key)
  {
    case 27:
      exit(0);
    break;

    case '0':
      resetSkinningToRest();
      break;

    case 9:
      fullScreen = 1 - fullScreen;
      if (fullScreen == 1)
        glutFullScreen();
      else {
        glutReshapeWindow(800, 600);
        glutPositionWindow(5, 150);
      }
      break;

    case '\\':
      camera->Reset();
    break;

    case '=':
      curJointID++;
      if (curJointID >= fk->getNumJoints())
        curJointID = -1;
    break;

    case 'a':
      showAxes = !showAxes;
      break;

    case 'w':
      showWireframe = !showWireframe;
      break;

    case 'e':
      showObject = !showObject;
      break;

    case 's':
      renderSkeleton = !renderSkeleton;
      break;

    default:
      break;
  }
}

static void specialKeysFunc(int key, int x, int y)
{
  switch (key)
  {
    case GLUT_KEY_LEFT:
      camera->MoveFocusRight(0.1 * fabs(camera->GetRadius()));
      break;

    case GLUT_KEY_RIGHT:
      camera->MoveFocusRight(-0.1 * fabs(camera->GetRadius()));
      break;

    case GLUT_KEY_DOWN:
      camera->MoveFocusUp(0.1 * fabs(camera->GetRadius()));
      break;

    case GLUT_KEY_UP:
      camera->MoveFocusUp(-0.1 * fabs(camera->GetRadius()));
      break;
  }
}

static void mouseNoDrag(int x, int y)
{
  id.setMousePos(x,y);
  if (handleControl.isHandleSelected())
  {
    Vec3d worldPos(0.0);
    GLubyte stencilValue;
    float zValue;
    unprojectPointFromScreen(x,y, &worldPos[0], &stencilValue, &zValue);

    if (stencilValue == 1)
    {
      handleControl.setMousePosition(worldPos);
    }
  }
}

static void mouseDrag(int x, int y)
{
  int mouseDeltaX = x-id.getMousePosX();
  int mouseDeltaY = y-id.getMousePosY();

  id.setMousePos(x,y);

  // we moved the camera...
  if (id.rightMouseButtonDown())
  { 
    // right mouse button handles camera rotations
    double scale = 0.2;
    if(id.shiftPressed()) scale *= 0.1;
    camera->MoveRight(scale * mouseDeltaX);
    camera->MoveUp(scale * mouseDeltaY);
  }

  if (id.middleMouseButtonDown() || (id.altPressed() && id.leftMouseButtonDown()))
  { 
    // middle mouse button (or ALT + left mouse button) handles camera translations
    double scale = 0.2 * modelRadius;
    if(id.shiftPressed()) scale *= 0.1;
    camera->ZoomIn(scale * mouseDeltaY);
  }
}

static void mouseButtonActivity(int button, int state, int x, int y)
{
  id.setButton(button, state);
  switch (button)
  {
    case GLUT_LEFT_BUTTON:
    {
      Vec3d clickedPosition(0.0);
      GLubyte stencilValue;
      float zValue = 0.0f;
      unprojectPointFromScreen(x,y, &clickedPosition[0], &stencilValue, &zValue);

      if (id.leftMouseButtonDown())
      {
        if (stencilValue == 0)
        {
          cout << "Clicked on empty space." << endl;
          selectedVertex = -1;
          return;
        }
        MinValueIndex vi;
        for(size_t i = 0; i < mesh->getNumVertices(); i++)
        {
          vi.update(len2(clickedPosition - mesh->getPosition(i)), i);
        }
        selectedVertex = vi.index;
        cout << "Clicked on vertex " << vi.index << endl;

        if (fk->getNumJoints() > 0)
        {
          MinValueIndex vi;
          for(int i = 0; i < fk->getNumJoints(); i++)
          {
            vi.update(len2(fk->getJointGlobalPosition(i) - clickedPosition), i);
          }
          assert(vi.index >= 0);

          if (vi.index != curJointID)
          {
            curJointID = vi.index;
            cout << "select joint ID " << curJointID << ", #joints " << fk->getNumJoints() << endl;
          }
        }
      }

      auto getClosestHandle = [&]() -> int
      {
        MinValueIndex vi;
        for(size_t handleID = 0; handleID < IKJointIDs.size(); handleID++)
        {
          vi.update(len2(clickedPosition - fk->getJointGlobalPosition(IKJointIDs[handleID])), handleID);
        }
        return vi.index;
      };
      auto addOrRemoveHandle = [&]()
      {
        return make_pair(-1, false);
      };
      handleControl.setMouseButtonActivity(id.leftMouseButtonDown(), stencilValue == 1, false,
          clickedPosition, zValue, getClosestHandle, addOrRemoveHandle);

      break;
    }

    case GLUT_MIDDLE_BUTTON:
      break;

    case GLUT_RIGHT_BUTTON:
      break;
  }
}

static void initialize()
{
  // initialize random number generator
  srand(time(nullptr));

  // detect the OpenGL version being used
  printf("GL_VENDOR: %s\n",glGetString(GL_VENDOR));
  printf("GL_RENDERER: %s\n",glGetString(GL_RENDERER));
  printf("GL_VERSION: %s\n",glGetString(GL_VERSION));

  mesh = new ObjMesh(meshFilename);
  meshDeformable = new SceneObjectDeformable(mesh, false);

  if (meshDeformable->HasTextures())
  {
    meshDeformable->EnableTextures();
    meshDeformable->SetUpTextures(SceneObject::MODULATE, SceneObject::NOMIPMAP);
  }
  meshDeformable->BuildNeighboringStructure();
  meshDeformable->BuildNormals();
  //  meshDeformable->BuildDisplayList();

  // ---------------------------------------------------
  // joint initialization
  // ---------------------------------------------------

  if (IKJointIDs.size() == 0)
  {
    cout << "No IK handles specified in the config file" << endl;
    exit(0);
  }

  assert(jointRestTransformsFilename.size() > 0 && jointWeightsFilename.size() > 0);
  skinning = new Skinning(meshDeformable->Getn(), meshDeformable->GetVertexRestPositions(), jointWeightsFilename);
  fk = new FK(jointHierarchyFilename, jointRestTransformsFilename);

  skinning->appliedSkinningMethod = SKINNINGMETHOD;

  // ---------------------------------------------------
  // Setting up Adol-c
  // ---------------------------------------------------
  ik = new IK(IKJointIDs.size(), IKJointIDs.data(), fk);
  ik->appliedIKMethod = IKMETHOD;
  IKJointPos.resize(IKJointIDs.size());
  for(size_t i = 0; i < IKJointIDs.size(); i++)
  {
    IKJointPos[i] = fk->getJointGlobalPosition(IKJointIDs[i]);
  }

  // ---------------------------------------------------
  // rendering setup
  // ---------------------------------------------------

  double cameraUp[3] = {0,1,0};

  Vec3d cameraFocus;
  Vec3d bmin, bmax;
  mesh->computeBoundingBox();
  mesh->getCubicBoundingBox(1.0, &bmin, &bmax);
  modelCenter = (bmin + bmax) / 2.0;
  modelRadius = mesh->getDiameter() / 2;

  // compute the size of the shape for getting a proper renderedLocalAxisLength
  double localAxisLength = modelRadius / 5.0;
  skeletonRenderer = new SkeletonRenderer(fk, localAxisLength);
  cout << "Finished joint initialization" << endl;

  double cameraRadius = 0;
  cameraFocus = modelCenter;
  cameraRadius = modelRadius * 2.5;
  zNear = cameraRadius * 0.01;
  zFar = cameraRadius * 100.0;

  double cameraPhi = 270.0;
  double cameraTheta = 0;
  camera = new SphericalCamera(cameraRadius,
      1.0 * cameraPhi / 360 * (2.0 * PI),
      1.0 * cameraTheta / 360 * (2.0 * PI),
      &cameraFocus[0], cameraUp, 0.05);

  //  lighting = new Lighting;
  //  lighting->SetLightBox(&bmin[0], &bmax[0]);
  //  lighting->SetAllLightsEnabled(false);
  //  lighting->SetLightEnabled(2, true);
  //  lighting->SetLightEnabled(3, true);
  //  lighting->SetLightEnabled(6, true);
  //  lighting->SetLightEnabled(7, true);
  //  lighting->SetAllLightsIntensity(allLightsIntensity);

  cameraLighting = new CameraLighting;
  cameraLighting->SetLightIntensity(allLightsIntensity);

  //   clear to white
  glClearColor(256.0 / 256, 256.0 / 256, 256.0 / 256, 0.0);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_STENCIL_TEST);
  glShadeModel(GL_SMOOTH);
  glEnable(GL_POLYGON_SMOOTH);
  glEnable(GL_LINE_SMOOTH);
  printf ("Initialization complete.\n");
  return;
}

#define ADD_CONFIG(v) configFile.addOptionOptional(#v, &v, v)
static void initConfigurations()
{
  ConfigFile configFile;

  ADD_CONFIG(allLightsIntensity);
  ADD_CONFIG(screenshotBaseName);
  ADD_CONFIG(meshFilename);

  // Maya data needs jointHierarchyFilename, jointRestTransformsFilename and jointWeightsFilename
  ADD_CONFIG(jointHierarchyFilename);
  ADD_CONFIG(jointRestTransformsFilename);
  ADD_CONFIG(jointWeightsFilename);
  ADD_CONFIG(IKJointIDs);

  // parse the configuration file
  if (configFile.parseOptions(configFilename.c_str()) != 0)
  {
    printf("Error parsing options.\n");
    exit(1);
  }

  // The config variables have now been loaded with their specified values.
  // Informatively print the variables (with assigned values) that were just parsed.
  configFile.printOptions();
}

int main (int argc, char ** argv)
{
  int numFixedArgs = 4;
  if ( argc < numFixedArgs )
  {
    cout << "Renders an obj mesh on the screen." << endl;
    cout << "Usage: " << argv[0] << " <configFilename>" << " <Inverse Kinematics Method>" << " <Skinning Method>" << endl;
    cout << "  Inverse Kinematics Method:" << endl;
    cout << "    t: Tikhonov Regularization Inverse Kinematics Method" << endl;
    cout << "    p: Pseudo Inverse Method" << endl;
    cout << "    k: Tikhonov Regularization Inverse Kinematics Method with Kernel Trick" << endl;
    cout << "  Skinning Method:" << endl;
    cout << "    l: Linear Blend Skinning" << endl;
    cout << "    d: Dual Quaternion Skinning" << endl;
    cout << "Example: " << argv[0] << " skin.config " << "t " << "l" << endl;
    return 0;
  }
 
  configFilename = argv[1];
  char * IKMethodString = argv[2];
  if (IKMethodString[0] == 'p')
    IKMETHOD = PseudoInverse;
  else if (IKMethodString[0] == 't')
    IKMETHOD = Tikhonov;
  else
    IKMETHOD = KernelTrickOnTikhonov;
  
  char* SkinningMethodString = argv[3];
  if (SkinningMethodString[0] == 'd')
    SKINNINGMETHOD = DualQuaternionSkinning;
  else
    SKINNINGMETHOD = LinearBlendSkinning;

  initConfigurations();

  glutInit(&argc,argv);
  glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL | GLUT_MULTISAMPLE);

  windowWidth = 800;
  windowHeight = 800;

  glutInitWindowSize (windowWidth,windowHeight);
  glutInitWindowPosition (0,0);
  windowID = glutCreateWindow ("IK viewer");
  if (fullScreen==1)
    glutFullScreen();

  #ifdef __APPLE__
    // This is needed on recent Mac OS X versions to correctly display the window.
    glutReshapeWindow(windowWidth - 1, windowHeight - 1);
  #endif

  initialize();

  // Initialize the WebSocket server
  websocketpp::server<websocketpp::config::asio> server;
  server.set_access_channels(websocketpp::log::alevel::all);
  server.clear_access_channels(websocketpp::log::alevel::frame_payload);
  server.set_message_handler(std::bind(&on_message, &server, std::placeholders::_1, std::placeholders::_2));

  // Start the WebSocket server in a separate thread
  std::thread server_thread([&server]() {
      server.init_asio();
      server.listen(9002);
      server.start_accept();
      server.run();
      });

  // callbacks
  glutDisplayFunc(displayFunction);
  glutMotionFunc(mouseDrag);
  glutPassiveMotionFunc(mouseNoDrag);
  glutIdleFunc(idleFunction);
  glutKeyboardFunc(keyboardFunc);
  glutSpecialFunc(specialKeysFunc);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouseButtonActivity);

  reshape(windowWidth,windowHeight);
  glutMainLoop();

  // When stop the server and close the application,
  // call server.stop() and then join the server_thread:
  server.stop();
  server_thread.join();

  return(0);
}

