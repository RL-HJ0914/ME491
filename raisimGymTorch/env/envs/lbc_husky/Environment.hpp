#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include <time.h>

/// gc_ = x,y,z positions
///       w,x,y,z quaternion
///       w1,w2,w3,w4 wheel angles

/// gv_ = x,y,z linear velocities
///       w_x,w_y,w_z angular velocities in the world frame
///       s1,s2,s3,s4 wheel speed

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// set the logger for debugging
    raisim::RaiSimMsg::setFatalCallback([](){throw;});

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add robot
    husky_ = world_->addArticulatedSystem(resourceDir_ + "/husky/husky.urdf");
    husky_->setName("husky");
    husky_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);

    /// add heightmap
    raisim::TerrainProperties terrainProperties;
    terrainProperties.frequency = 0.2;
    terrainProperties.zScale = 2.0;
    terrainProperties.xSize = 70.0;
    terrainProperties.ySize = 70.0;
    terrainProperties.xSamples = 70;
    terrainProperties.ySamples = 70;
    terrainProperties.fractalOctaves = 3;
    terrainProperties.fractalLacunarity = 2.0;
    terrainProperties.fractalGain = 0.25;

    std::unique_ptr<raisim::TerrainGenerator> genPtr = std::make_unique<raisim::TerrainGenerator>(terrainProperties);
    std::vector<double> heightVec = genPtr->generatePerlinFractalTerrain();

    /// add obstacles
    for (int i = 0; i < 70; i += GRIDSIZE) {
      for (int j = (i % (GRIDSIZE * GRIDSIZE)) * 2 / GRIDSIZE; j < 70; j += GRIDSIZE) {
        poles_.emplace_back(Eigen::Vector2d{i - 35.0, j - 35.0});
        heightVec[i*70 + j] += 1.;
      }
    }
    heightMap_ = world_->addHeightMap(terrainProperties.xSamples,
                                      terrainProperties.ySamples,
                                      terrainProperties.xSize,
                                      terrainProperties.xSize,
                                      0.,
                                      0.,
                                      heightVec);


    /// get robot data
    gcDim_ = husky_->getGeneralizedCoordinateDim();
    gvDim_ = husky_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    lidarData.setZero(SCANSIZE);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    genForce_.setZero(gvDim_);
    torque4_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 17 + SCANSIZE;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(50.);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(husky_);

      /// lidar points visualization
      for(int i = 0; i < SCANSIZE; i++)
        scans.push_back(server_->addVisualBox("box" + std::to_string(i), 0.1, 0.1, 0.1, 1, 0, 0));
      origin.push_back(server_->addVisualCylinder("origin_cylinder",0.05,0.6,0,1,1,0.7));
    }
    for(auto& rw: rewards_.getStdMap()) {
      stepDataTag_.push_back(rw.first);
    }

    stepData_.resize(stepDataTag_.size());
  }

  void init() final { }

  void reset() final {
    {
      double xPos, yPos;

      do {
        int i = int((uniDist_(gen_) * .5 + 0.5) * poles_.size());
        xPos = poles_[i](0) + GRIDSIZE / 2.;
        yPos = poles_[i](1) + GRIDSIZE / 2.;
      } while(xPos > 30 || yPos > 30 || xPos < 5 || yPos < 5);

      double height = heightMap_->getHeight(xPos, yPos);
      gc_init_.head(3) << xPos, yPos, height + 0.2;
      husky_->setState(gc_init_, gv_init_);
    }
    updateObservation();
  }

  const std::vector<std::string>& getStepDataTag() {
    return stepDataTag_;
  }

  const Eigen::VectorXf& getStepData() {
    int i = 0;
    for(auto& rw: rewards_.getStdMap()){
      stepData_[i] = rw.second;
      i++;
    }
    return stepData_;
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    torque4_ = action.cast<double>();
    torque4_ = torque4_.cwiseProduct(actionStd_);
    torque4_ += actionMean_;
    genForce_.tail(nJoints_) = torque4_;


    husky_->setGeneralizedForce(genForce_);

    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();

    rewards_.record("goal", std::max(gc_.head<2>().norm(),3.));
    rewards_.record("ori",reward_ori);
    rewards_.record("near",reward_near);
    rewards_.record("vel",reward_vel);
    rewards_.record("reward_avoid",reward_avoid);
    return rewards_.sum();

  }

  void updateObservation() {
//    auto visCylinder = server_->addVisualCylinder("v_cylinder", 1, 1, 0, 1, 0, 1);
//    visCylinder->setPosition(0,2,0);
    husky_->getState(gc_, gv_);
    raisim::Vec<3> lidarPos; raisim::Mat<3,3> lidarOri;
    husky_->getFramePosition("imu_joint", lidarPos);
    husky_->getFrameOrientation("imu_joint", lidarOri);

    raisim::Vec<4> quat;

    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    goal_ori << -gc_(0),-gc_(1);
    goal_ori /= (goal_ori.norm()+1e-8);
    robot_ori << rot(0,0),rot(1,0);
    robot_ori /= (robot_ori.norm()+1e-8);

    reward_ori= 1-acos(goal_ori.dot(robot_ori))/M_PI;

    vel_robotframe= rot.e().transpose()*gv_.head(3);
    reward_vel=vel_robotframe(0);

    if (near_zero()) reward_near=1;
    else reward_near=0;

    Eigen::Vector3d direction;
//    const double scanWidth = 2. * M_PI; // original
    const double scanWidth = 20 * M_PI/180;

    for (int j = 0; j < SCANSIZE; j++) {
//      const double yaw = j * M_PI / SCANSIZE * scanWidth - scanWidth * 0.5 * M_PI; // original
      const double yaw = -(SCANSIZE-1)/2*scanWidth + scanWidth*j;
//      direction = {-cos(yaw), -sin(yaw), -0.1 * M_PI}; // original one
      direction = {cos(yaw), sin(yaw), 0};
      direction *= 1. / direction.norm();
      Eigen::Vector3d rayDirection = lidarOri.e() * direction; // original one
      rayDirection(2)=-0.1*M_PI;

      auto &col = world_->rayTest(lidarPos.e(), rayDirection, SCANSIZE);
      if (col.size() > 0) {
        lidarData[j] = (col[0].getPosition() - lidarPos.e()).norm();
        if (visualizable_)
          scans[j]->setPosition(col[0].getPosition());
      } else { // lidar reflect 신호가 잡히지 않는 경우
        lidarData[j] = 20;
        if (visualizable_)
          scans[j]->setPosition({0,0,100});
      }
    }
    min_distance=lidarData.minCoeff();

    reward_avoid=fmin(2*(min_distance-0.6),0);
//    std::cout<<"lidardata:"<<lidarData.transpose()<<std::endl;
    obDouble_ << gc_.head(7), gv_, lidarData;

    //make cylinder indicating direction to origin
    Eigen::Vector3d cylinder_pos,x_,y_,z_;
    Eigen::Matrix3d cylinder_rot;
    Vec<4> cylinder_quat ;
    Eigen::Vector4d cylinder_quat_;
    z_ << -gc_(0),-gc_(1),0;
    x_<< 0,0,1;
    z_/=(z_.norm()+1e-10);
    x_/=(x_.norm()+1e-10);
    y_= crossProduct(z_,x_);
    cylinder_rot.block(0,0,3,1)=x_;
    cylinder_rot.block(0,1,3,1)=y_;
    cylinder_rot.block(0,2,3,1)=z_;
    raisim::rotMatToQuat(cylinder_rot,cylinder_quat);
    cylinder_pos<<gc_(0),gc_(1),gc_(2)+0.5;
    cylinder_pos+=z_*1.2;
    cylinder_quat_ ={cylinder_quat(0),cylinder_quat(1),cylinder_quat(2),cylinder_quat(3)};
//    cylinder_quat_={0.816,0.41,0,0.41};
    cylinder_quat_/=cylinder_quat_.norm();

    if(visualizable_) {
      origin[0]->setPosition(cylinder_pos);
      origin[0]->setOrientation(cylinder_quat_);
    }
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    ob = obDouble_.cast<float>();
  }
  double leakyrelu(double angle){
    double x= angle- angle_threshold;
    if (x>0) return -5; // tilted too much
    else return 0;
  }

  Eigen::Vector3d crossProduct(Eigen::Vector3d A, Eigen::Vector3d B){
    Eigen::Vector3d C;
    C[0]=A[1]*B[2]-A[2]*B[1];
    C[1]=-(A[0]*B[2]-A[2]*B[0]);
    C[2]=A[0]*B[1]-A[1]*B[0];
    return C;
  }
  bool isTerminalState(float& terminalReward) final {
    if (rot(2,2)<0) {
      terminalReward = -100;
      return true;
    }
    return false;


  }

  float notCompleted() {
    if (gc_.head(2).norm() < 2)
      return 0.f;
    else
      return 1.f;
  }

  bool near_zero() {
    if (gc_.head(2).norm() < 2)
      return true;
    else
      return false;
  }

  void curriculumUpdate() { // curriculumFactor_은 1에서 0으로 감
    curriculumFactor_*=0.995;
  };

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* husky_;
  raisim::HeightMap* heightMap_;
  raisim::Mat<3,3> rot;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, genForce_, torque4_, lidarData;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector2d goal_ori, robot_ori;
  Eigen::Vector3d vel_robotframe;
  double min_distance;
  double reward_ori, reward_near,reward_vel, reward_avoid;
  double angle_threshold=50 ; //degree
  double curriculumFactor_ =1;
  std::vector<Eigen::Vector2d> poles_;
  int SCANSIZE = 9; // original = 20
  int GRIDSIZE = 6;
  std::vector<raisim::Visuals *> scans,origin;  // for visualization

  Eigen::VectorXf stepData_;
  std::vector<std::string> stepDataTag_;

  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
};

thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(-1., 1.);
}

