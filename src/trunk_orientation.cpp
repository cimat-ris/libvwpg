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

#include <boost/date_time/posix_time/posix_time.hpp>

#include <Eigen/Dense>

#include "common.h"
#include "trunk_orientation.h"
#include <fstream>

//******************************************************************************
//
//                         RobotOrientationFollower
//
//******************************************************************************

RobotTrunkOrientationFollower::RobotTrunkOrientationFollower(const std::vector<FPTYPE> &angles, const int N) :
    angles_(angles.size(),1), N_(N), current_iteration_(0), trunk_angles_(N_,1)
{
    for (int i=0;i<angles.size();i++) {
        angles_(i,0) = angles[i];
    }
    trunk_angles_ = angles_.block(current_iteration_, 0, N_, 1);
}

RobotTrunkOrientationFollower::~RobotTrunkOrientationFollower()
{
    // nothing to do.
}

void RobotTrunkOrientationFollower::UpdateReference(FPTYPE reference_angle)
{
    ++current_iteration_;
    trunk_angles_ = angles_.block(current_iteration_, 0, N_, 1);
}

const Matrix_t &RobotTrunkOrientationFollower::GetFeetOrientation()
{
    return trunk_angles_;
}


const Matrix_t &RobotTrunkOrientationFollower::GetTrunkOrientation()
{
    return trunk_angles_;
}

void RobotTrunkOrientationFollower::ComputeRobotNextOrientation() {
    //nothing to do.
}

void RobotTrunkOrientationFollower::SetSupportFootOrientation(FPTYPE support_orientation_angle)
{
    //Nothing to do.
}

FPTYPE RobotTrunkOrientationFollower::GetSupportFootOrientation()
{
    // Should not be used
    return 0.0;
}

FPTYPE RobotTrunkOrientationFollower::GetCoMAngularSpeed()
{
    // Should not be used
    return 0.0;
}

FPTYPE RobotTrunkOrientationFollower::GetCoMAngularAcceleration()
{
    // Should not be used
    return 0.0;
}
//******************************************************************************
//
//                         RobotOrientationOptimizer 
//
//******************************************************************************

RobotTrunkOrientationOptimizer::RobotTrunkOrientationOptimizer(int N, FPTYPE T, int m, FPTYPE alpha, FPTYPE beta, FPTYPE gamma) :
    N_(N),
    trunk_orientation_model_(N, T, m, alpha, beta, gamma),
    target_angle_(0),
    qp_solver_(dynamic_cast<QProblemInterface*>(&trunk_orientation_model_))
{
}


RobotTrunkOrientationOptimizer::~RobotTrunkOrientationOptimizer()
{
    // nothing to do
}

void RobotTrunkOrientationOptimizer::UpdateReference(FPTYPE reference_angle)
{
    trunk_orientation_model_.SetReference(reference_angle);
    target_angle_ = reference_angle;
}

void RobotTrunkOrientationOptimizer::ComputeRobotNextOrientation()
{
    try {
        auto qp_start_time = boost::posix_time::microsec_clock::local_time();
        qp_solver_.SolveProblem();
        auto qp_end_time   = boost::posix_time::microsec_clock::local_time();
        auto duration = qp_end_time - qp_start_time;
        common::logString("[qp_time_orientation_lineal]: time=" + boost::lexical_cast<std::string>(duration.total_microseconds()) + " microseconds");
    } catch (...) {
        std::cerr << "ERROR: Solving orientation QP Problem " << std::endl;
        throw;
    }
    common::logString("[orientation_objective_function]: " + boost::lexical_cast<std::string>(qp_solver_.GetObjectiveFunctionValue()));
    // Applies the angle solver
    Matrix_t jerks = qp_solver_.GetSolution();
    const Matrix_t &Pps = trunk_orientation_model_.GetPps_();
    const Matrix_t &Ppu = trunk_orientation_model_.GetPpu_();
    Vector3D_t trunk_state_ = trunk_orientation_model_.GetTrunkState_();
    Vector3D_t feet_state_  = trunk_orientation_model_.GetFootState_();

    // Deduce the predicted values of com/feet orientations
    Matrix_t degresCoM  = Pps * trunk_state_ + Ppu * jerks.block(0,0,N_,1);
    Matrix_t degresFeet = Pps * feet_state_  + Ppu * jerks.block(N_,0, N_,1);
    trunk_orientation_model_.SetFootAngles(degresFeet);
    trunk_orientation_model_.SetTrunkAngles(degresCoM);

    // Applies the first computed jerk to prepare the model for the next step
    PrepareDataForNextOptimization(jerks);
}

void RobotTrunkOrientationOptimizer::PrepareDataForNextOptimization(const Matrix_t &jerks)
{
    Vector3D_t trunk_state_ = trunk_orientation_model_.GetTrunkState_();
    Vector3D_t feet_state_  = trunk_orientation_model_.GetFootState_();
    const Matrix_t &A     = trunk_orientation_model_.GetA_();
    const Matrix_t &b     = trunk_orientation_model_.GetB_();
    feet_state_     = A * feet_state_  + b * jerks(N_);
    trunk_state_    = A * trunk_state_ + b * jerks(0);
    feet_state_(0)  = 0.0;
    trunk_state_(0) = 0.0;
    trunk_orientation_model_.SetFootState(feet_state_);
    trunk_orientation_model_.SetTrunkState(trunk_state_);
    trunk_orientation_model_.orientation_step_generator_.UpdateSequence();
    trunk_orientation_model_.SetThetaV_();
    trunk_orientation_model_.SetPpuk_();
}
const Matrix_t &RobotTrunkOrientationOptimizer::GetFeetOrientation()
{
    return trunk_orientation_model_.GetFootAngles();
}


const Matrix_t &RobotTrunkOrientationOptimizer::GetTrunkOrientation()
{
    return trunk_orientation_model_.GetTrunkAngles();
}

void RobotTrunkOrientationOptimizer::SetSupportFootOrientation(FPTYPE support_orientation_angle)
{
    trunk_orientation_model_.SetSupportFootOrientation_(support_orientation_angle);
}

FPTYPE RobotTrunkOrientationOptimizer::GetSupportFootOrientation()
{
    return  trunk_orientation_model_.GetSupportFootOrientation_();
}

FPTYPE RobotTrunkOrientationOptimizer::GetCoMAngularSpeed()
{
    return trunk_orientation_model_.GetTrunkState_()[1];
}

FPTYPE RobotTrunkOrientationOptimizer::GetCoMAngularAcceleration()
{
    return trunk_orientation_model_.GetTrunkState_()[2];
}
//******************************************************************************
//
//                          RobotTrunkOrientationModel
//
//******************************************************************************

// public:

RobotTrunkOrientationModel::RobotTrunkOrientationModel(int N, FPTYPE T, int m, FPTYPE alpha, FPTYPE beta, FPTYPE gamma)
    : N_(N), A_(3, 3), B_(3, 1), Pps_(N, 3), Ppu_(N, N), Ppuk_(N,N), thetaV_(N,1), alpha_(alpha), beta_(beta), gamma_(gamma),
      trunk_state_(0, 0, 0), foot_state_(0, 0, 0), orientation_step_generator_(N,m,Foot::RIGHT_FOOT,2,7),     
      trunk_angles_(N_,1), foot_angles_(N,1)

{
    InitA(T);
    InitB(T);
    InitPps(T);
    InitPpu(T);
    InitPpuk();
    InitThetaV();
    trunk_angles_.setZero();
    foot_angles_.setZero();
}

Matrix_t RobotTrunkOrientationModel::GetH()
{
    /*
       H = [\alpha_R*I + \beta*Ppu^T*Ppu                0
                       0                  \alpha_R*I + \gamma_R*Ppuk^T*Ppuk]*/
    Matrix_t H(2 * N_, 2 * N_);
    H.setZero();
    const Matrix_t A = alpha_ * Matrix_t::Identity(N_, N_);
    H.block(0, 0, N_, N_) = A + beta_ * (Ppu_.transpose() * Ppu_);
    const Matrix_t &Ppuk = GetPpuk_();
    H.block(N_, N_, N_, N_) = A + gamma_ * (Ppuk.transpose()*Ppuk);
    return H;
}

Matrix_t RobotTrunkOrientationModel::Getg()
{
    /*
     * pk = [\beta_R*Ppu^T*(Pps*theta_state - theta_ref)
     *       \gamma_R*Ppu^T(Pps*foot_state_- theta_ref) ]*/
    Matrix_t g(2* N_, 1);
    g.setZero();
    Matrix_t reference = RefAnglesInterpolation(reference_angle_);
    g.block( 0, 0, N_, 1) = beta_ * Ppu_.transpose() * (Pps_ * trunk_state_ - reference);
    const Matrix_t &Ppuk   = GetPpuk_();
    const Matrix_t &thetaV = GetThetaV_();
    g.block(N_, 0, N_, 1) = gamma_ * Ppuk.transpose() * (thetaV - reference);
    return g;
}

Matrix_t RobotTrunkOrientationModel::GetA()
{
    Matrix_t A(N_, 2 * N_);
    A.setZero();
    A.block(0, 0, N_, N_) = Ppu_ ;
    const Matrix_t &Ppuk = GetPpuk_();
    A.block(0, N_, N_, N_) = - Ppuk;
    return A;
}

Matrix_t RobotTrunkOrientationModel::GetlbA()
{
    Matrix_t lbA(N_, 1);
    lbA.setZero();
    const Matrix_t &thetaV = GetThetaV_();
    lbA = -MAX_THETA_ * Matrix_t::Ones(N_, 1) - (Pps_ * trunk_state_) + thetaV;
    return lbA;
}

Matrix_t RobotTrunkOrientationModel::GetubA()
{
    Matrix_t ubA(N_, 1);
    ubA.setZero();
    const Matrix_t &thetaV = GetThetaV_();
    ubA = MAX_THETA_ * Matrix_t::Ones(N_, 1) - (Pps_ * trunk_state_) + thetaV;
    return ubA;
}

Matrix_t RobotTrunkOrientationModel::Getlb()
{
    Matrix_t empty_matrix;
    return empty_matrix;
}

Matrix_t RobotTrunkOrientationModel::Getub()
{
    Matrix_t empty_matrix;
    return empty_matrix;
}

int RobotTrunkOrientationModel::GetNumberOfVariables() 
{
    return 2 * N_;
}

int RobotTrunkOrientationModel::GetNumberOfConstraints() 
{
    return N_;
}

const Matrix_t &RobotTrunkOrientationModel::GetPpu_()
{
    return Ppu_;
}

const Matrix_t &RobotTrunkOrientationModel::GetPps_()
{
    return Pps_;
}

const Matrix_t &RobotTrunkOrientationModel::GetA_()
{
    return A_;
}

const Matrix_t &RobotTrunkOrientationModel::GetB_()
{
    return B_;
}

const Vector3D_t &RobotTrunkOrientationModel::GetFootState_()
{
    return foot_state_;
}

const Vector3D_t &RobotTrunkOrientationModel::GetTrunkState_()
{
    return trunk_state_;
}

void RobotTrunkOrientationModel::SetFootState(const Vector3D_t &foot_state_new)
{
    foot_state_ = foot_state_new;
}

void RobotTrunkOrientationModel::SetTrunkState(const Vector3D_t &trunk_state_new)
{
    trunk_state_ = trunk_state_new;
}

RobotTrunkOrientationModel::~RobotTrunkOrientationModel()
{
    // nothing to do
}

// private
void RobotTrunkOrientationModel::InitA(const FPTYPE T)
{
    A_.setZero();
    A_ << 1.0,   T, std::pow(T, 2.0) / 2.0,
          0.0, 1.0,                      T,
          0.0, 0.0,                    1.0;
}

void RobotTrunkOrientationModel::InitB(const FPTYPE T)
{
    B_.setZero();
    B_ << std::pow(T, 3.0) / 6.0,
          std::pow(T, 2.0) / 2.0,
          T;
}

void RobotTrunkOrientationModel::InitPps(const FPTYPE T)
{
    Pps_.setZero();
    for (int i=0; i<N_; i++) {
        Pps_(i, 0) = 1.0;
        Pps_(i, 1) = (i + 1) * T;
        Pps_(i, 2) = std::pow(((i + 1) * T), 2)/2;
    }
}

void RobotTrunkOrientationModel::InitPpu(const FPTYPE T)
{
    Ppu_.setZero();
    for (int i = 0; i < N_; i++)
        for (int j = 0; j <= i; j++)
            Ppu_(i, j) = (3*((i-j+1)<<1) - 3*(i-j+1) + 1) * (std::pow(T, 3) / 6.0);    
}

void RobotTrunkOrientationModel::SetReference(const FPTYPE reference_angle)
{
    reference_angle_ = reference_angle;
}

Matrix_t RobotTrunkOrientationModel::RefAnglesInterpolation(const FPTYPE ref_angle)
{
    Matrix_t references(N_,1);
    references.setZero();
    FPTYPE orientation;
    orientation = ref_angle > 0 ? std::min(MAX_ORIENTATION_, ref_angle) : std::max(-MAX_ORIENTATION_, ref_angle);
    const FPTYPE delta = orientation / ((FPTYPE)N_ - 1);

    for (int i = 0; i < N_; i++)
         references(i) = (i+1) * delta;

    return references;
}

void RobotTrunkOrientationModel::SetFootAngles(const Matrix_t &degresFeet)
{
        foot_angles_  = degresFeet;
}

void RobotTrunkOrientationModel::SetTrunkAngles(const Matrix_t &degresCoM) 
{
        trunk_angles_ = degresCoM;
}

const Matrix_t &RobotTrunkOrientationModel::GetFootAngles()
{
    return foot_angles_;
}

const Matrix_t &RobotTrunkOrientationModel::GetTrunkAngles()
{
    return trunk_angles_;
}

void RobotTrunkOrientationModel::PrintFootAngles()
{
    std::cout << "\n " ;
    for(int i=0;i<N_;i++)
        std::cout <<  foot_angles_(i,0) << std::endl;
}

void RobotTrunkOrientationModel::PrintTrunkAngles()
{
    std::cout << "\n " ;
    for(int i=0;i<N_;i++)
        std::cout <<  trunk_angles_(i,0) << std::endl;
}

void RobotTrunkOrientationModel::SetPpuk_()
{
    //For the flying foot.
    Matrix_t Uc = orientation_step_generator_.GetCurrent_U_Matrix();
    Matrix_t Uf = orientation_step_generator_.GetFuture_U_Matrix();
    unsigned short int s1 = orientation_step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
    unsigned short int s2 = orientation_step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));

    Ppuk_.setZero();
    Ppuk_.block(0,0,s1,s1) = Ppu_.block(0,0,s1,s1); //block 1x1
    Ppuk_.block(s2,0,N_-s2,s1) = Pps_.block(0,0,N_-s2,1)*Ppu_.block(s1-1,0,1,s1); //block 1x3
    Ppuk_.block(s1,s1,s2-s1,s2-s1) = Ppu_.block(0,0,s2-s1,s2-s1); //block 2x2
    Ppuk_.block(s2,s2,N_-s2,N_-s2) = Ppu_.block(0,0,N_-s2,N_-s2); // block 3x3
}

void RobotTrunkOrientationModel::SetThetaV_()
{
    //For the flying foot.
    Matrix_t Uc = orientation_step_generator_.GetCurrent_U_Matrix();
    Matrix_t Uf = orientation_step_generator_.GetFuture_U_Matrix();
    unsigned short int s1 = orientation_step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
    unsigned short int s2 = orientation_step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));

    thetaV_.setZero();
    thetaV_.block(0,0,s1,1) = Pps_.block(0,0,s1,3)*foot_state_; //
    thetaV_.block(s1,0,s2-s1,1) = support_foot_orientation_*Pps_.block(0,0,s2-s1,1);
    // This part  prevents numerical errors
    Matrix_t theta_prev = (Pps_.block(s1-1,0,1,3)*foot_state_);
    thetaV_.block(s2,0,N_-s2,1) = theta_prev(0,0)*Pps_.block(0,0,N_-s2,1);
}

const Matrix_t &RobotTrunkOrientationModel::GetPpuk_()
{
    return Ppuk_;
}

const Matrix_t &RobotTrunkOrientationModel::GetThetaV_()
{
    return thetaV_;
}

void RobotTrunkOrientationModel::SetSupportFootOrientation_(const FPTYPE angle)
{
    this->support_foot_orientation_ = angle;
}

void RobotTrunkOrientationModel::InitPpuk()
{
    SetPpuk_();
}

void RobotTrunkOrientationModel::InitThetaV()
{
    SetThetaV_();
}

Matrix_t RobotTrunkOrientationModel::RefAnglesFootInterpolation(const FPTYPE ref_angle)
{
    Matrix_t references(N_,1);
    references.setZero();

    //Reference for the flying foot.
    Matrix_t Uc = orientation_step_generator_.GetCurrent_U_Matrix();
    Matrix_t Uf = orientation_step_generator_.GetFuture_U_Matrix();
    unsigned short int s1 = orientation_step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
    unsigned short int s2 = orientation_step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));
    FPTYPE orientation;
    orientation = ref_angle > 0 ? std::min(MAX_ORIENTATION_, ref_angle) : std::max(-MAX_ORIENTATION_, ref_angle);
    const FPTYPE delta2 = orientation / ((FPTYPE)(N_-(s2-s1)) - 1);

    for(int i=0; i<s1;i++)
        references(i) = (i+1)*delta2;

    for(int i=s1; i<s2;i++)
        references(i) = (i+1-s1)*delta2;

    for(int i=s2; i<N_;i++)
        references(i) = (i-(s2-s1))*delta2;

    return references;
}

FPTYPE RobotTrunkOrientationModel::GetSupportFootOrientation_()
{
    return support_foot_orientation_;
}