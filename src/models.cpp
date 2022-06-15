/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <cmath>
#include <vector>
#include <memory>
#include <iostream>

#include <Eigen/Dense>

#include "models.h"
#include "common.h"


/*******************************************************************************
 *
 *                                 DynamicModel
 *
 ******************************************************************************/
// public:
UnidimensionalIntegrator::UnidimensionalIntegrator(const FPTYPE &T, const FPTYPE &position,  const FPTYPE &speed, const FPTYPE &acceleration):
    A_(3, 3),
    B_(3, 1),
    state_vector_(3, 1)
{
    initialize_A_(T);
    initialize_B_(T);
    initialize_state_vector_(position, speed, acceleration);
}

// private:
void UnidimensionalIntegrator::initialize_A_(const FPTYPE &T)
{
    A_.setZero();
    A_ << 1.0,   T, std::pow(T, 2.0) / 2.0,
          0.0, 1.0,                      T,
          0.0, 0.0,                    1.0;
}

void UnidimensionalIntegrator::initialize_B_(const FPTYPE &T)
{
    B_.setZero();
    B_ << std::pow(T, 3.0) / 6.0,
          std::pow(T, 2.0) / 2.0,
          T;
}


void UnidimensionalIntegrator::initialize_state_vector_(const FPTYPE &position, const FPTYPE &speed, const FPTYPE &acceleration)
{
    state_vector_.setZero();
    state_vector_ << position, speed, acceleration;
}

//
FlyingFootUnidimensionalIntegrator::FlyingFootUnidimensionalIntegrator(const FPTYPE &T, const FPTYPE &position,  const FPTYPE &speed, const FPTYPE &acceleration) : UnidimensionalIntegrator(T,position,speed,acceleration),support_foot_orientation_(0.0) {}


// public:
UnidimensionalCartModel::UnidimensionalCartModel(const FPTYPE &T, const FPTYPE &CoM_height, const FPTYPE &zmp_position,
                                                 const FPTYPE &position,  const FPTYPE &speed, const FPTYPE &acceleration):
    UnidimensionalIntegrator(T,position,speed,acceleration),
    c_(1, 3),
    zmp_position_(zmp_position) {
    initialize_c_(CoM_height);
}

void UnidimensionalCartModel::UpdateState(const FPTYPE &jerk)
{
    state_vector_ = A_*state_vector_ + B_*jerk;
    zmp_position_ = (c_*state_vector_)(0);
}

void UnidimensionalCartModel::initialize_c_(const FPTYPE &CoM_height)
{
    c_.setZero();
    c_ << 1.0, 0.0, -CoM_height/common::g;
}

/*******************************************************************************
 *
 *                              RobotDynamicModel
 *
 ******************************************************************************/

RobotDynamicModel:: RobotDynamicModel(const FPTYPE T, const FPTYPE CoM_height,
                                      const FPTYPE zmp_x_position, const FPTYPE zmp_y_position,
                                      const FPTYPE x_position,  const FPTYPE x_speed, const FPTYPE x_acceleration,
                                      const FPTYPE y_position,  const FPTYPE y_speed, const FPTYPE y_acceleration,
                                      const FPTYPE tcom_position,  const FPTYPE tcom_speed, const FPTYPE tcom_acceleration,
                                      const FPTYPE tfoot_position,  const FPTYPE tfoot_speed, const FPTYPE tfoot_acceleration):
    x_axis_model_(T, CoM_height, zmp_x_position, x_position, x_speed, x_acceleration),
    y_axis_model_(T, CoM_height, zmp_y_position, y_position, y_speed, y_acceleration),
    tcom_model_(T,tcom_position, tcom_speed, tcom_acceleration),
    tfoot_model_(T,tfoot_position, tfoot_speed, tfoot_acceleration)
{

}

// Positions only
void RobotDynamicModel::UpdateState(FPTYPE x_jerk, FPTYPE y_jerk)
{
    x_axis_model_.UpdateState(x_jerk);
    y_axis_model_.UpdateState(y_jerk);
}

// Positions and angles
void RobotDynamicModel::UpdateState(FPTYPE x_jerk, FPTYPE y_jerk, FPTYPE tcom_jerk, FPTYPE tfoot_jerk)
{
    x_axis_model_.UpdateState(x_jerk);
    y_axis_model_.UpdateState(y_jerk);
    tcom_model_.UpdateState(tcom_jerk);
    tfoot_model_.UpdateState(tfoot_jerk);
}

void RobotDynamicModel::SetCoM_X_Position(FPTYPE position)
{
    x_axis_model_.SetPosition(position);
}

void RobotDynamicModel::SetCoM_Y_Position(FPTYPE position)
{
    y_axis_model_.SetPosition(position);
}

void RobotDynamicModel::SetCoM_X_Speed(FPTYPE speed)
{
    x_axis_model_.SetVelocity(speed);   
}

void RobotDynamicModel::SetCoM_Y_Speed(FPTYPE speed)
{
    y_axis_model_.SetVelocity(speed);   
}

void RobotDynamicModel::SetCoM_angle_Position(FPTYPE position)
{
    tcom_model_.SetPosition(position);
}

void RobotDynamicModel::SetFoot_angle_Position(FPTYPE position)
{
    tfoot_model_.SetPosition(position);
}

void RobotDynamicModel::ResetFlyingFootState() {
    tfoot_model_.ResetState();
}

void RobotDynamicModel::ResetTranslationStateVector(const FPTYPE angle)
{

    x_axis_model_.SetPosition(0.0);
    y_axis_model_.SetPosition(0.0);

    Matrix_t state_vector_ = x_axis_model_.GetStateVector();
    Matrix_t mat_rot_(2,2);mat_rot_.setZero();
    // TODO: is the sign ok?
    mat_rot_ << std::cos(angle), -std::sin(angle),
               std::sin(angle), std::cos(angle);
    
    Matrix_t v_vector_(2,1); v_vector_.setZero();
    v_vector_ << x_axis_model_.GetSpeed(), y_axis_model_.GetSpeed();
    Matrix_t nv_vector_ = mat_rot_*v_vector_;
    x_axis_model_.SetVelocity(nv_vector_(0));
    y_axis_model_.SetVelocity(nv_vector_(1));

    Matrix_t a_vector_(2,1); a_vector_.setZero();
    a_vector_ << x_axis_model_.GetAcceleration(), y_axis_model_.GetAcceleration();    
    Matrix_t na_vector_ = mat_rot_*a_vector_;
    x_axis_model_.SetAcceleration(na_vector_(0));
    y_axis_model_.SetAcceleration(na_vector_(1));
}

void RobotDynamicModel::ResetOrientationStateVector()
{
    const FPTYPE curcom  = tcom_model_  .GetPosition();
    const FPTYPE curffoot= tfoot_model_ .GetPosition();
    const FPTYPE cursfoot= tfoot_model_ .GetSupport();
    tcom_model_ .SetPosition(0.0);
    tfoot_model_.SetPosition(curffoot-curcom);
    tfoot_model_.SetSupport(cursfoot-curcom);
}

/*******************************************************************************
 *
 *                                 MPCModel
 *
 ******************************************************************************/

// public:

MPCModel::MPCModel(const int N, const FPTYPE T, const FPTYPE CoM_height, unsigned short int s1, unsigned short int s2):
    Pps_(N, 3),
    Ppu_(N, N),
    Pvs_(N, 3),
    Pvu_(N, N),
    Pas_(N, 3),
    Pau_(N, N),
    Pzs_(N, 3),
    Pzu_(N, N),
    Ppuk_(N,N),
    theta_V(N,1)
{

    InitializePps(N, T);
    InitializePpu(N, T);

    InitializePvs(N, T);
    InitializePvu(N, T);

    InitializePas(N, T);
    InitializePau(N, T);

    InitializePzs(N, T, CoM_height);
    InitializePzu(N, T, CoM_height);

    InitializePpuk(s1,s2,N);
    InitializeThetaV(s1,s2,N,0.0);

}

void MPCModel::SetPpuk(unsigned short int s1, unsigned short int s2, const int N)
{
    Ppuk_.setZero();
    Ppuk_.block(0,0,s1,s1)        = Ppu_.block(0,0,s1,s1); //block 1x1
    Ppuk_.block(s2,0,N-s2,s1)     = Pps_.block(0,0,N-s2,1)*Ppu_.block(s1-1,0,1,s1); //block 1x3
    Ppuk_.block(s1,s1,s2-s1,s2-s1)= Ppu_.block(0,0,s2-s1,s2-s1); //block 2x2
    Ppuk_.block(s2,s2,N-s2,N-s2)  = Ppu_.block(0,0,N-s2,N-s2); // block 3x3
}

void MPCModel::SetThetaV(unsigned short int s1, unsigned short int s2, const int N,
                         const FPTYPE support_foot_orientation_, 
                         const Matrix_t &foot_state_)
{

    theta_V.setZero();
    theta_V.block(0,0,s1,1) = Pps_.block(0,0,s1,3)*foot_state_; //
    theta_V.block(s1,0,s2-s1,1) = support_foot_orientation_*Pps_.block(0,0,s2-s1,1);
    // This part  prevents numerical errors
    Matrix_t theta_prev = (Pps_.block(s1-1,0,1,3)*foot_state_);
    theta_V.block(s2,0,N-s2,1) = theta_prev(0,0)*Pps_.block(0,0,N-s2,1);

}

// private:

void MPCModel::InitializePpuk(unsigned short int s1, unsigned short int s2, const int N)
{
    SetPpuk(s1,s2,N);
}

void MPCModel::InitializeThetaV(unsigned short int s1, unsigned short int s2, const int N,
                                const FPTYPE support_foot_orientation_)
{
    Matrix_t foot_init_state_vector(3,1);
    foot_init_state_vector.setZero();
    SetThetaV(s1,s2,N,support_foot_orientation_,foot_init_state_vector);
}

void MPCModel::InitializePps(const int N, const FPTYPE T)
{
    Pps_.setZero();

    for (int i=0; i<N; i++) {
        Pps_(i, 0) = 1.0;
        Pps_(i, 1) = (i + 1) * T;
        Pps_(i, 2) = std::pow(((i + 1) * T), 2)/2;
    }
}

void  MPCModel::InitializePvs(const int N, const FPTYPE T)
{
    Pvs_.setZero();
    for (int i=0; i<N; i++) {
        Pvs_(i, 1) = 1.0;
        Pvs_(i, 2) = (i + 1) * T;
    }
}

void  MPCModel::InitializePas(const int N, const FPTYPE T)
{
    Pas_.setZero();
    for (int i=0; i<N; i++)
        Pas_(i, 2) = 1.0;
}

void MPCModel::InitializePzs(const int N, const FPTYPE T, FPTYPE h)
{
    Pzs_.setZero();
    Pzs_ = Pps_;
    for (int i = 0; i < N; i++)
        Pzs_(i, 2) -= h / common::g;
}

void MPCModel::InitializePpu(const int N, const FPTYPE T)
{
    Ppu_.setZero();
    for (int i = 0; i < N; i++)
        for (int j = 0; j <= i; j++)
	        Ppu_(i, j) = (3*std::pow(i-j+1, 2.0) - 3*(i-j+1) + 1) * (std::pow(T, 3) / 6.0);
}


void MPCModel::InitializePvu(const int N, const FPTYPE T)
{
    Pvu_.setZero();
    for (int i = 0; i < N; i++)
        for (int j = 0; j <= i; j++)
            Pvu_(i, j) = (1 + 2*((i-j+1) - 1)) * (std::pow(T, 2) / 2.0);
}

void MPCModel::InitializePau(const int N, const FPTYPE T)
{
    Pau_.setZero();
    for (int i=0; i<N; i++)
        for (int j=0; j<=i; j++)
            Pau_(i, j) = T;
}


void MPCModel::InitializePzu(const int N, const FPTYPE T, FPTYPE h)
{
    Pzu_.setZero();
    Pzu_ = Ppu_;
    for (int i = 0; i < N; i++)
        for (int j = 0; j <= i; j++)
            Pzu_(i, j) -= T * h / common::g;

}

/*******************************************************************************
 *
 *                              MPCHomographyModel 
 *
 ******************************************************************************/

// public:
MPCLinearHomography::MPCLinearHomography(std::string type, int N) : MPCLinearVisualConstraint(type,N) {

    if (type == "h33") {
        vconstraint_entry_hxx_ = boost::shared_ptr<HomographyEntryInterface>(new HomographyH33());
        return;
    }

    if (type == "h13") {
        vconstraint_entry_hxx_ =  boost::shared_ptr<HomographyEntryInterface>(new HomographyH13());
        return;
    }

    if (type == "h11") {
        vconstraint_entry_hxx_ =  boost::shared_ptr<HomographyEntryInterface>(new HomographyH11());
        return;
    }

    if (type == "h31") {
        vconstraint_entry_hxx_ =  boost::shared_ptr<HomographyEntryInterface>(new HomographyH31());
        return;
    }

    if (type == "h12") {
        vconstraint_entry_hxx_ =  boost::shared_ptr<HomographyEntryInterface>(new HomographyH12());
        return;
    }

    if (type == "h32") {
        vconstraint_entry_hxx_ =  boost::shared_ptr<HomographyEntryInterface>(new HomographyH32());
        return;
    }
  throw std::domain_error("ERROR: Invalid type for an MPCLinearHomography");
}

/*******************************************************************************
 *
 *                              MPCNonlinearHomographyModel
 *
 ******************************************************************************/

// public:
MPCNonLinearHomography::MPCNonLinearHomography(std::string type, int N) : MPCVisualConstraint(type,N) {

    if (type == "h33") {
        vconstraint_entry_hxx_ = boost::shared_ptr<HomographyEntryInterface>(new HomographyH33());
        return;
    }

    if (type == "h13") {
        vconstraint_entry_hxx_ =  boost::shared_ptr<HomographyEntryInterface>(new HomographyH13());
        return;
    }

    if (type == "h11") {
        vconstraint_entry_hxx_ =  boost::shared_ptr<HomographyEntryInterface>(new HomographyH11());
        return;
    }

    if (type == "h31") {
        vconstraint_entry_hxx_ =  boost::shared_ptr<HomographyEntryInterface>(new HomographyH31());
        return;
    }

    if (type == "h12") {
        vconstraint_entry_hxx_ =  boost::shared_ptr<HomographyEntryInterface>(new HomographyH12());
        return;
    }

    if (type == "h32") {
        vconstraint_entry_hxx_ =  boost::shared_ptr<HomographyEntryInterface>(new HomographyH32());
        return;
    }
    throw std::domain_error("ERROR: Invalid type for an MPCNonLinearHomography");
}

/*******************************************************************************
 *
 *                              MPCLinearEssential 
 *
 ******************************************************************************/

// public:
MPCLinearEssential::MPCLinearEssential(std::string type, int N) : MPCLinearVisualConstraint(type,N) {

  if (type == "e12") {
    vconstraint_entry_hxx_ = boost::shared_ptr<EssentialEntryInterface>(new EssentialE12());
    return;
  }
  if (type == "e21") {
    vconstraint_entry_hxx_ = boost::shared_ptr<EssentialEntryInterface>(new EssentialE21());
    return;
  }
  if (type == "e23") {
    vconstraint_entry_hxx_ = boost::shared_ptr<EssentialEntryInterface>(new EssentialE23());
    return;
  }
  if (type == "e32") {
    vconstraint_entry_hxx_ = boost::shared_ptr<EssentialEntryInterface>(new EssentialE32());
    return;
  }

  throw std::domain_error("ERROR: Invalid type for an MPCLinearEssential");
}

/*******************************************************************************
 *
 *                              MPCNonLinearEssential 
 *
 ******************************************************************************/

// public:
MPCNonLinearEssential::MPCNonLinearEssential(std::string type, int N) : MPCVisualConstraint(type,N) {

  if (type == "e12") {
    vconstraint_entry_hxx_ = boost::shared_ptr<EssentialEntryInterface>(new EssentialE12());
    return;
  }
  if (type == "e21") {
    vconstraint_entry_hxx_ = boost::shared_ptr<EssentialEntryInterface>(new EssentialE21());
    return;
  }
  if (type == "e23") {
    vconstraint_entry_hxx_ = boost::shared_ptr<EssentialEntryInterface>(new EssentialE23());
    return;
  }
  if (type == "e32") {
    vconstraint_entry_hxx_ = boost::shared_ptr<EssentialEntryInterface>(new EssentialE32());
    return;
  }
  throw std::domain_error("ERROR: Invalid type for an MPCNonLinearEssential");
}


/*******************************************************************************
 *                              Robot Parameters 
 ******************************************************************************/

RobotModelInterface* InstantiateRobotModel(std::string name)
{
    if (name == "hrp") 
        return new HRPModel();

    if (name == "nao")
        return new NaoModel();

    if (name == "jvrc")
        return new JVRCModel();

    std::cout << "WARNING: defaulting to HRP Model" << std::endl;
    return new HRPModel();
}
