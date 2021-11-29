#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

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
        poles_.emplace_back(Eigen::Vector2d{1.01449*j - 35.0, 1.01449*i - 35.0});
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
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    genForce_.setZero(gvDim_);
    torque4_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 19 + SCANSIZE;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(20.);

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
      origin.push_back(server_->addVisualCylinder("origin_cylinder",0.08,0.6,0,1,0,1));
      origin.push_back(server_->addVisualCylinder("horn1",0.2,80,0,0,1,0.05));
      origin.push_back(server_->addVisualBox("origin",0.2,0.2,0.2,1,1,0.5,1));
    }
    for(auto& rw: rewards_.getStdMap()) {
      stepDataTag_.push_back(rw.first);
    }

    stepData_.resize(stepDataTag_.size());

  }

  void init() final {
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
    rewards_.record("goal", -gc_.head<2>().norm()-0*collide_with_horn());
    rewards_.record("ori", reward_ori);
    rewards_.record("vel",reward_vel);
    rewards_.record("near",reward_near);

    return rewards_.sum();
  }

  void updateObservation() {
    husky_->getState(gc_, gv_);

    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    goal_ori<< -gc_(0), -gc_(1), 0;
    robot_ori << -rot(0,0), -rot(1,0), 0; // -x direction of robot
    robot_vel_ori << gv_(0), gv_(1),0;
    goal_ori/=goal_ori.norm();
    robot_ori/=robot_ori.norm();
    robot_vel_ori/=(robot_vel_ori.norm()+1e-8);
    get_nearest_horn(gc_.head(2));
    raisim::Vec<3> lidarPos; raisim::Mat<3,3> lidarOri;
    husky_->getFramePosition("imu_joint", lidarPos);
    husky_->getFrameOrientation("imu_joint", lidarOri);

    reward_ori=goal_ori.dot(robot_ori);
//    reward_vel= -(rot.e().transpose()*gv_.head(3))(0);// -x direction velocity
    reward_vel= goal_ori.dot(gv_.head(3))/5;
    reward_near=1-notCompleted();//near이면 1 아니면 0

    visualize_cylinder();

    // if using lidar
    if (SCANSIZE > 0) {
      Eigen::Vector3d direction;
      const double scanWidth = 10 * M_PI/180;
      Eigen::VectorXd lidarData(SCANSIZE);
      for (int j = 0; j < SCANSIZE; j++) {
        const double yaw = -(SCANSIZE - 1) / 2 * scanWidth + scanWidth * j;
        direction = {cos(yaw), sin(yaw), -0.05 * M_PI};
        direction *= 1. / direction.norm();
        const Eigen::Vector3d rayDirection = lidarOri.e() * direction;
        auto &col = world_->rayTest(lidarPos.e(), rayDirection, 20);
        if (col.size() > 0) {
          lidarData[j] = (col[0].getPosition() - lidarPos.e()).norm();
          if (visualizable_)
            scans[j]->setPosition(col[0].getPosition());
        } else {
          lidarData[j] = 20;
          if (visualizable_)
            scans[j]->setPosition({0, 0, 100});
        }
      }
      obDouble_ << gc_.head(7), gv_,  lidarData, dist_to_horn, angle_to_horn;
    }
    else obDouble_ << gc_.head(7), gv_, dist_to_horn, angle_to_horn;

  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    ob = obDouble_.cast<float>();
  }
  void visualize_cylinder(){
    z_ = goal_ori;
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
      origin[1]->setPosition(Eigen::Vector3d{pos_nearest_horn[0],pos_nearest_horn[1],0});
      origin[1]->setOrientation(Eigen::Vector4d{1,0,0,0});
      origin[2]->setPosition(Eigen::Vector3d{0,0,2});
    }

  }
  void get_nearest_horn(Eigen::Vector2d gc){ // 전방에 보이는 90도 안의 범위에 있는 horn들 중 가장 가까운 것을 고른다.
    min_distance=1e10;
    min_index=0;
    for (int i=0; i<poles_.size(); i++){
      dist_to_horn=(gc-poles_[i]).norm();
      vec_to_horn<<poles_[i]-gc,0;
      vec_to_horn/=vec_to_horn.norm();

      if (dist_to_horn < min_distance and vec_to_horn.dot(robot_vel_ori) > cos(60.0/180*M_PI)){
        min_index=i;
        min_distance=dist_to_horn;
      }
    }
    dist_to_horn=(gc-poles_[min_index]).norm();
    vec_to_horn<<poles_[min_index]-gc,0;
    vec_to_horn/=vec_to_horn.norm();
//    std::cout<<"robot_vel_ori: "<<robot_vel_ori<<std::endl;
//    std::cout<<"vec_to_horn: "<<vec_to_horn<<std::endl;
    angle_to_horn= asin(crossProduct(robot_vel_ori,vec_to_horn)(2)); //진행방향의 왼쪽에 horn이 있으면 + value
    pos_nearest_horn= poles_[min_index];
  }
  double collide_with_horn(){
    if(dist_to_horn < danger_radius) return 1;
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
//    if(rot(2,2)<0) {
//      terminalReward = -100.;
//      return true;
//    }
    return false;

  }

  double reward_shape(double x1, double x2, double m){ // 1 before x1, m after x2
    if (iter < x1) return 1;
    else if (iter > x2) return m;
    else return 1+(m-1)/(x2-x1)*(iter-x1);
  }

  float notCompleted() { // 0 for arrived
    if (gc_.head(2).norm() < 2)
      return 0.f;
    else
      return 1.f;
  }

  void curriculumUpdate() {
    iter+=1;
  };

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  int iter=0;
  raisim::ArticulatedSystem* husky_;
  raisim::HeightMap* heightMap_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, genForce_, torque4_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  raisim::Vec<4> quat;
  raisim::Mat<3,3> rot;
  Eigen::Vector3d robot_ori,robot_vel_ori, goal_ori;
  std::vector<Eigen::Vector2d> poles_;
  int SCANSIZE = 0;
  int GRIDSIZE = 6;
  std::vector<raisim::Visuals *> scans;  // for visualization
  std::vector<raisim::Visuals *> origin;  // for visualization
  Eigen::Vector3d cylinder_pos,x_,y_,z_;
  Eigen::Matrix3d cylinder_rot;
  Vec<4> cylinder_quat ;
  Eigen::Vector4d cylinder_quat_;

  double reward_ori, reward_vel, reward_near;
  Eigen::VectorXf stepData_;
  std::vector<std::string> stepDataTag_;

  double danger_radius=2.1, dist_to_horn=0, angle_to_horn=0;
  Eigen::Vector2d pos_nearest_horn;
  Eigen::Vector3d vec_to_horn;
  int min_index;
  double min_distance;

  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
};


thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(-1., 1.);
}

