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
#include <map>
#include <string>
#include <numeric>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "formulations.h"
#include "vision_utils.h"

/*******************************************************************************
 *
 *                                   HerdtBase
 *
 ******************************************************************************/

HerdBase::HerdBase(const boost::property_tree::ptree &parameters):
    FormulationBase(parameters),
    iterations_(parameters.get<int>("simulation.iterations")),
    world_next_support_foot_orientation_(0.0),
    world_support_foot_orientation_(0.0),
    world_flying_foot_orientation_(0.0),
    world_com_orientation_(0.0),
    alpha_(parameters.get<FPTYPE>("qp.alpha")),
    beta_(parameters.get<FPTYPE>("qp.beta")),
    gamma_(parameters.get<FPTYPE>("qp.gamma"))
{

    rotation_controller_ = boost::shared_ptr<RobotTrunkOrientationOptimizer>(new RobotTrunkOrientationOptimizer(N_, parameters.get<FPTYPE>("simulation.T"),m_,parameters.get<FPTYPE>("qp.alphaR"),parameters.get<FPTYPE>("qp.betaR"),parameters.get<FPTYPE>("qp.gammaR")));
    world_T_com_ = Eigen::Translation<FPTYPE, 3>(0, 0, robot_physical_parameters_->com_height());
    // Initial world-to-foot (used for displaying results)
    world_T_foot_ = Eigen::Translation<FPTYPE , 3>(0, 0, 0) * Eigen::AngleAxis<FPTYPE>(0.0, Point3D_t::UnitZ());
    world_T_next_foot_ = world_T_foot_;
    // Initial
    com_T_foot_ = Eigen::Translation<FPTYPE , 3>(0, 0, -robot_physical_parameters_->com_height()) * Eigen::AngleAxis<FPTYPE>(0.0, Point3D_t::UnitZ());
    com_T_next_foot_  = com_T_foot_;
}

HerdBase::HerdBase(const std::string nameFileParameters):
        FormulationBase(nameFileParameters),
        iterations_(parameters_.get<int>("simulation.iterations")),
        world_next_support_foot_orientation_(0.0),
        world_support_foot_orientation_(0.0),
        world_flying_foot_orientation_(0.0),
        world_com_orientation_(0.0),
        alpha_(parameters_.get<FPTYPE>("qp.alpha")),
        beta_(parameters_.get<FPTYPE>("qp.beta")),
        gamma_(parameters_.get<FPTYPE>("qp.gamma"))
{

    common::logDesiredPosition(current_iteration_,parameters_.get<FPTYPE>("reference.camera_position0_x"), parameters_.get<FPTYPE>("reference.camera_position0_y"),
                               parameters_.get<FPTYPE>("reference.orientation0" ) );
    rotation_controller_ = boost::shared_ptr<RobotTrunkOrientationOptimizer>(new RobotTrunkOrientationOptimizer(N_, parameters_.get<FPTYPE>("simulation.T"),m_,parameters_.get<FPTYPE>("qp.alphaR"),parameters_.get<FPTYPE>("qp.betaR"),parameters_.get<FPTYPE>("qp.gammaR")));


    world_T_com_ = Eigen::Translation<FPTYPE, 3>(0, 0, robot_physical_parameters_->com_height());
    // Initial world-to-foot (used for displaying results)
    world_T_foot_ = Eigen::Translation<FPTYPE , 3>(0, 0, 0) * Eigen::AngleAxis<FPTYPE>(0.0, Point3D_t::UnitZ());
    world_T_next_foot_ = world_T_foot_;
    // Initial
    com_T_foot_ = Eigen::Translation<FPTYPE , 3>(0, 0, -robot_physical_parameters_->com_height()) * Eigen::AngleAxis<FPTYPE>(0.0, Point3D_t::UnitZ());
    com_T_next_foot_  = com_T_foot_;

}

Matrix_t HerdBase::GetH()
{
    Matrix_t H(2*(N_+m_), 2*(N_+m_));
    Matrix_t Q_prime(N_+m_, N_+m_);

    Matrix_t Pvu = mpc_model_.get_Pvu();
    Matrix_t Pzu = mpc_model_.get_Pzu();
    Matrix_t U_future = step_generator_.GetFuture_U_Matrix();

    Q_prime.setZero();
    Q_prime.block(0, 0, N_, N_) = alpha_*Matrix_t::Identity(N_, N_) + beta_*Pvu.transpose()*Pvu + gamma_*Pzu.transpose()*Pzu;
    Q_prime.block(0, N_, N_, m_) = -gamma_*Pzu.transpose()*U_future;
    Q_prime.block(N_, 0, m_, N_) = -gamma_*U_future.transpose()*Pzu;
    Q_prime.block(N_, N_, m_, m_) = gamma_*U_future.transpose()*U_future;

    H.setZero();
    H.block(0, 0, N_+m_, N_+m_) = Q_prime;
    H.block(N_+m_, N_+m_, N_+m_, N_+m_) = Q_prime;

    return H;
}

void HerdBase::Update(const Matrix_t &solution)
{

    // Update simulation based on the previous solution. ``Moves'' the dynamical system according to the computed solution.
    UpdateSimulation(solution);

    // Prepare data for the next optimization step
    PrepareDataForNextOptimization();

}


void HerdBase::UpdateSimulation(const Matrix_t &solution)
{

    FPTYPE x_jerk = solution(0, 0);
    FPTYPE y_jerk = solution(N_+m_, 0);
    FPTYPE footorientation = 0.0;
    FPTYPE dfootorientation= 0.0;
    FPTYPE comorientation  = 0.0;
    FPTYPE nextFootOrientation = 0.0;


    // Update the internal parameters of the step generator
    // May toggle the support foot
    step_generator_.UpdateSequence();


    dynamic_model_.UpdateState(x_jerk, y_jerk);
    // Flying foot and CoM orientations (in the previous CoM frame)
    footorientation     = rotation_controller_->GetFeetOrientation()(0,0);
    comorientation      = rotation_controller_->GetTrunkOrientation()(0,0);
    nextFootOrientation = rotation_controller_->GetFeetOrientation()(7,0);


    // Update the flying foot orientation: it is equal to the orientation of the previous CoM frame + footorientation
    world_flying_foot_orientation_       = world_com_orientation_ + footorientation;


    if (!step_generator_.IsSameSupportFoot()) {
        common::logString(": change support foot");
        X(support_foot_position_) = solution(N_, 0);
        Y(support_foot_position_) = solution(N_ + m_ + N_, 0);
        // Determines the transform from the new support foot to the previous CoM frame
        // The orientation is the one the currently flying foot has (but that will be reset)
        com_T_foot_ =   Eigen::Translation<FPTYPE , 3>(X(support_foot_position_), Y(support_foot_position_), -robot_physical_parameters_->com_height()) *
                        Eigen::AngleAxis<FPTYPE>(footorientation, Point3D_t::UnitZ());
        // world_T_com_ is the transformation from the previous CoM frame to the World
        // We deduce the transformation from the new support foot to the World
        world_T_foot_ = world_T_com_ * com_T_foot_;
        // Log flying foot absolute orientation
        common::logSupportFootAngle(current_iteration_+1, world_support_foot_orientation_);
        common::logFlyingFootAngle(current_iteration_ +1, world_flying_foot_orientation_);
        world_flying_foot_orientation_  = world_support_foot_orientation_;
        // Support foot orientation is updated here and in PrepareDataForNextOptimization
        dynamic_model_.ResetFlyingFootState();

        // Deduce the new support foot world orientation from world_T_foot_
        world_support_foot_orientation_ = -std::atan2(world_T_foot_(0,1),world_T_foot_(0,0));

        com_T_next_foot_ =   Eigen::Translation<FPTYPE , 3>(X(support_foot_position_), Y(support_foot_position_), -robot_physical_parameters_->com_height()) *
                             Eigen::AngleAxis<FPTYPE>(nextFootOrientation, Point3D_t::UnitZ());
        // world_T_com_ is the transformation from the previous CoM frame to the World
        // We deduce the transformation from the new support foot to the World
        world_T_next_foot_ = world_T_com_ * com_T_next_foot_;
        world_next_support_foot_orientation_ = -std::atan2(world_T_next_foot_(0,1),world_T_next_foot_(0,0));

        rotation_controller_->SetSupportFootOrientation(footorientation);
    }

    // transformation matrices
    mk_T_mk1 = Eigen::Translation<FPTYPE, 3>(dynamic_model_.GetCoM_X_Position(), dynamic_model_.GetCoM_Y_Position(), 0.0) *
               Eigen::AngleAxis<FPTYPE>(comorientation,  Eigen::Vector3d::UnitZ());


    world_T_com_ = world_T_com_ * mk_T_mk1;
    // Update the world_com_orientation (for simulation display only)
    world_com_orientation_  += comorientation;

    ++current_iteration_;
}

void HerdBase::PrepareDataForNextOptimization()
{

    // New CoM origin is set to zero
    dynamic_model_.ResetTranslationStateVector(rotation_controller_->GetTrunkOrientation()(0,0));

    // Update support foot position which is now written in the frame mk
    AffineTransformation mk1_T_mk = mk_T_mk1.inverse();
    support_foot_position_ = mk1_T_mk * support_foot_position_;

    // Update the target angle to reach from the current essential matrix
    // Optimize the jerks to apply, update the rotation controller
    SolveOrientation();

}

void HerdBase::LogCurrentPredictions(const Matrix_t &solution) const {
    // TODO: Write this function
}

void HerdBase::LogCurrentResults(const Matrix_t &solution) const
{
    std::ostringstream oss;
    Point3D_t point(0.0,0.0,0.0);

    //point << dynamic_model_.GetCoM_X_Position(), dynamic_model_.GetCoM_Y_Position(), 0.0;
    point = world_T_com_ * point;
    common::logCoMPosition(current_iteration_, X(point), Y(point));
    // Log CoM absolute orientation
    common::logCoMAngle(current_iteration_,  world_com_orientation_);
    // Log flying foot absolute orientation
    common::logFlyingFootAngle(current_iteration_, world_flying_foot_orientation_);
    // Log support foot absolute orientation
    common::logSupportFootAngle(current_iteration_, world_support_foot_orientation_);
    // Log current speed (in local frame)
    common::logCoMSpeed(current_iteration_,
                        dynamic_model_.GetCoM_X_Speed(),
                        dynamic_model_.GetCoM_Y_Speed());
    // Log current acceleration (in local frame)
    common::logCoMAcceleration(current_iteration_,
                               dynamic_model_.GetCoM_X_Acceleration(),
                               dynamic_model_.GetCoM_Y_Acceleration());
    // No solution for this step
    if (!solution.rows())
        return;

    // log information
    Point3D_t zmp(dynamic_model_.GetZMP_X_Position(),
                  dynamic_model_.GetZMP_Y_Position(),
                  -robot_physical_parameters_->com_height());
    zmp = mk_T_mk1.inverse() * zmp;
    zmp = world_T_com_ * zmp;
    common::logZMPPosition(current_iteration_, X(zmp), Y(zmp));

    // Get the first step of the solution, the jerks in x, y, tcom and tfoot
    const FPTYPE &x_jerk = solution(0    , 0);
    const FPTYPE &y_jerk = solution(N_+m_, 0);
    common::logJerks(current_iteration_, x_jerk, y_jerk);

    FPTYPE comAngularSpeed = rotation_controller_->GetCoMAngularSpeed();
    FPTYPE comAngularAcceleration = rotation_controller_->GetCoMAngularAcceleration();
    common::logCoMAngularSpeed(current_iteration_,comAngularSpeed);
    common::logCoMAngularAcceleration(current_iteration_,comAngularAcceleration);

    // Log the predicted foot positions
    // TODO: is rotation_controller_->GetFeetOrientation() an absolute rotation or relative one?
    const Matrix_t &reference_orientation = rotation_controller_->GetFeetOrientation();
    for (int i=0; i<m_; i++) {
        Point3D_t point(solution(N_ + i, 0), solution(N_ + m_ + N_ + i, 0), -robot_physical_parameters_->com_height());
        point = world_T_com_ * point;
        common::logPredictedFootPosition(current_iteration_, X(point), Y(point), reference_orientation(m_,0));
    }
    // Log support foot position whenever there is a change on it
    if (!step_generator_.IsSameSupportFoot()) {
        Point3D_t point(support_foot_position_(0), support_foot_position_(1), -robot_physical_parameters_->com_height());
        point = world_T_com_ * point;
        common::logFootPosition(current_iteration_, X(point), Y(point), world_support_foot_orientation_ );
    }

}


//******************************************************************************
//
//                        QPHerdtSim
//
//******************************************************************************
Herdt::Herdt(const boost::property_tree::ptree &parameters):
    HerdBase(parameters),
    reference_com_x_speed_(common::ExpandRawRange(parameters.get<std::string>("reference.x_com_speed"))),
    reference_com_y_speed_(common::ExpandRawRange(parameters.get<std::string>("reference.y_com_speed"))),
    reference_angles_(common::ExpandRawRange(parameters.get<std::string>("reference.orientation")))
{
    std::cout << "reference_angles___: " << reference_angles_[0] << std::endl;
    if (reference_angles_.size() != iterations_) {
        std::cerr << "ERROR: Invalid size for reference orientation, expected "
                  << iterations_ << " found " << reference_angles_.size() << std::endl;
        throw std::exception();
    }
    
    if (reference_com_x_speed_.size() != iterations_) {
        std::cerr << "ERROR: Invalid size for reference CoM X speed, expected "
                  << iterations_ << " found " << reference_com_x_speed_.size() << std::endl;
        throw std::exception();
    } else {
        for (int i=0; i<N_; i++)
            reference_com_x_speed_.push_back(reference_com_x_speed_.back());
    }

    if (reference_com_y_speed_.size() != iterations_) {
        std::cerr << "ERROR: Invalid size for reference CoM Y speed, expected "
                  << iterations_ << " found " << reference_com_y_speed_.size() << std::endl;
        throw std::exception();
    } else {
        for (int i=0; i<N_; i++)
            reference_com_y_speed_.push_back(reference_com_y_speed_.back());
    }

    for (int i=0; i<reference_com_x_speed_.size(); i++) {
        auto x_speed = reference_com_x_speed_[i];
        auto y_speed = reference_com_y_speed_[i];
        reference_com_x_speed_[i] = x_speed;
        reference_com_y_speed_[i] = y_speed;
    }

    common::logString(": HerdtSim constructor OK");
}

Herdt::Herdt(const std::string nameFileParameters):
        HerdBase(nameFileParameters),
        reference_com_x_speed_(common::ExpandRawRange(parameters_.get<std::string>("reference.x_com_speed"))),
        reference_com_y_speed_(common::ExpandRawRange(parameters_.get<std::string>("reference.y_com_speed"))),
        reference_angles_(common::ExpandRawRange(parameters_.get<std::string>("reference.orientation")))
{
    common::logString(": HerdtSim constructor");

    if (reference_angles_.size() != iterations_) {
        std::cerr << "ERROR: Invalid size for reference orientation, expected "
                  << iterations_ << " found " << reference_angles_.size() << std::endl;
        throw std::exception();
    }

    if (reference_com_x_speed_.size() != iterations_) {
        std::cerr << "ERROR: Invalid size for reference CoM X speed, expected "
                  << iterations_ << " found " << reference_com_x_speed_.size() << std::endl;
        throw std::exception();
    } else {
        for (int i=0; i<N_; i++)
            reference_com_x_speed_.push_back(reference_com_x_speed_.back());
    }

    if (reference_com_y_speed_.size() != iterations_) {
        std::cerr << "ERROR: Invalid size for reference CoM Y speed, expected "
                  << iterations_ << " found " << reference_com_y_speed_.size() << std::endl;
        throw std::exception();
    } else {
        for (int i=0; i<N_; i++)
            reference_com_y_speed_.push_back(reference_com_y_speed_.back());
    }

    for (int i=0; i<reference_com_x_speed_.size(); i++) {
        auto x_speed = reference_com_x_speed_[i];
        auto y_speed = reference_com_y_speed_[i];
        reference_com_x_speed_[i] = x_speed;
        reference_com_y_speed_[i] = y_speed;
    }


    common::logString(": HerdtSim constructor OK");
}


Matrix_t Herdt::Getg()
{
    Matrix_t g(2*N_ + 2*m_, 1);

    Matrix_t Pvu = mpc_model_.get_Pvu();
    Matrix_t Pzu = mpc_model_.get_Pzu();
    Matrix_t Pzs = mpc_model_.get_Pzs();
    Matrix_t Pvs = mpc_model_.get_Pvs();

    Matrix_t U_future = step_generator_.GetFuture_U_Matrix();
    Matrix_t U_current = step_generator_.GetCurrent_U_Matrix();

    Matrix_t x_state = dynamic_model_.GetXStateVector();
    Matrix_t y_state = dynamic_model_.GetYStateVector();

    Matrix_t X_speed_ref(N_, 1);
    Matrix_t Y_speed_ref(N_, 1);

    for (int i=current_iteration_; i<current_iteration_+N_; i++) {
        X_speed_ref(i-current_iteration_, 0) = reference_com_x_speed_[i];
        Y_speed_ref(i-current_iteration_, 0) = reference_com_y_speed_[i];
    }

    g.setZero();
    g.block(0, 0, N_, 1) = beta_ * Pvu.transpose() * (Pvs * x_state - X_speed_ref) + gamma_ * Pzu.transpose() * (Pzs * x_state - U_current * X(support_foot_position_));
    g.block(N_, 0, m_, 1) = -gamma_ * U_future.transpose() * (Pzs * x_state - U_current * X(support_foot_position_));
    g.block(N_ + m_, 0, N_, 1) = beta_ * Pvu.transpose() * (Pvs * y_state - Y_speed_ref) + gamma_ * Pzu.transpose() * (Pzs * y_state - U_current * Y(support_foot_position_));
    g.block(N_ + m_ + N_, 0, m_, 1) = -gamma_ * U_future.transpose() * (Pzs * y_state - U_current * Y(support_foot_position_));

    return g;
}


void Herdt::SolveOrientation()
{
    rotation_controller_->UpdateReference(reference_angles_[current_iteration_]);
    // Performs the optimization for the next step
    rotation_controller_->ComputeRobotNextOrientation();
}


/*******************************************************************************
 *
 *                                   HerdtReal
 *
 ******************************************************************************/
HerdtReal::HerdtReal(const boost::property_tree::ptree &parameters, const FPTYPE x_speed_ref, const FPTYPE y_speed_ref, const FPTYPE theta_ref):
HerdBase(parameters)
{ 
    SetCurrentReferenceSpeed(x_speed_ref,y_speed_ref,theta_ref);
    common::logString(": HerdtReal constructor OK");
}

HerdtReal::HerdtReal(const std::string nameFileParameters, const FPTYPE x_speed_ref, const FPTYPE y_speed_ref, const FPTYPE theta_ref):
HerdBase(nameFileParameters)
{ 
    SetCurrentReferenceSpeed(x_speed_ref,y_speed_ref,theta_ref);
    common::logString(": HerdtReal constructor OK");
}

void HerdtReal::SetCurrentReferenceSpeed(const FPTYPE x_speed_ref, const FPTYPE y_speed_ref, const FPTYPE theta_ref)
{
    x_speed_ref_ = x_speed_ref;
    y_speed_ref_ = y_speed_ref;
    theta_ref_   = theta_ref;
}


Matrix_t HerdtReal::Getg()
{
    
    Matrix_t g(2*N_ + 2*m_, 1);

    Matrix_t Pvu = mpc_model_.get_Pvu();
    Matrix_t Pzu = mpc_model_.get_Pzu();
    Matrix_t Pzs = mpc_model_.get_Pzs();
    Matrix_t Pvs = mpc_model_.get_Pvs();

    Matrix_t U_future = step_generator_.GetFuture_U_Matrix();
    Matrix_t U_current = step_generator_.GetCurrent_U_Matrix();

    Matrix_t x_state = dynamic_model_.GetXStateVector();
    Matrix_t y_state = dynamic_model_.GetYStateVector();

    Matrix_t X_speed_ref(N_, 1);
    Matrix_t Y_speed_ref(N_, 1);


    for (int i=0; i<N_; i++) {
        X_speed_ref(i, 0) = x_speed_ref_;
        Y_speed_ref(i, 0) = y_speed_ref_;
    }

    g.setZero();
    g.block(0, 0, N_, 1) = beta_ * Pvu.transpose() * (Pvs * x_state - X_speed_ref) + gamma_ * Pzu.transpose() * (Pzs * x_state - U_current * X(support_foot_position_));
    g.block(N_, 0, m_, 1) = -gamma_ * U_future.transpose() * (Pzs * x_state - U_current * X(support_foot_position_));
    g.block(N_ + m_, 0, N_, 1) = beta_ * Pvu.transpose() * (Pvs * y_state - Y_speed_ref) + gamma_ * Pzu.transpose() * (Pzs * y_state - U_current * Y(support_foot_position_));
    g.block(N_ + m_ + N_, 0, m_, 1) = -gamma_ * U_future.transpose() * (Pzs * y_state - U_current * Y(support_foot_position_));

    return g;

}

void HerdtReal::SolveOrientation()
{
    rotation_controller_->UpdateReference(theta_ref_);
    // Performs the optimization for the next step
    rotation_controller_->ComputeRobotNextOrientation();
}