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

#include <vector>
#include <memory>

#include <Eigen/Dense>

#include "feet.h"
#include "common.h"
#include "models.h"
#include "qp.h"
#include "common_formulation.h"
#include "trunk_orientation.h"

InitParameters::InitParameters(const std::string nameFileParameters)
{
    parameters_ = common::LoadParameters(nameFileParameters);
}

InitParameters::InitParameters(const boost::property_tree::ptree &parameters)
{
    parameters_ = parameters;
}

// Constructor
FormulationBase::FormulationBase(const boost::property_tree::ptree &parameters): 
    InitParameters(parameters),
    current_iteration_(0),
    N_(parameters.get<int>("simulation.N")),
    m_(parameters.get<int>("simulation.m")),
    logPredictions_(false),
    robot_physical_parameters_(InstantiateRobotModel(parameters.get<std::string>("simulation.robot"))),

    support_foot_position_(parameters.get<FPTYPE>("initial_values.foot_x_position"),
                           parameters.get<FPTYPE>("initial_values.foot_y_position"),
                           -robot_physical_parameters_->com_height()),

    dynamic_model_(parameters.get<FPTYPE>("simulation.T"), robot_physical_parameters_->com_height(),
                   parameters.get<FPTYPE>("initial_values.foot_x_position"), 
                   parameters.get<FPTYPE>("initial_values.foot_y_position"),
                   parameters.get<FPTYPE>("initial_values.foot_x_position"), 0.0, 0.0,
                   parameters.get<FPTYPE>("initial_values.foot_y_position"), 0.0, 0.0),

    robot_restrictions_(robot_physical_parameters_.get(),
                        parameters.get<int>("simulation.N"),
                        parameters.get<int>("simulation.m")),

    step_generator_(parameters.get<int>("simulation.N"),
                    parameters.get<int>("simulation.m"),
                    Foot::RIGHT_FOOT,
                    parameters.get<FPTYPE>("simulation.double_support_lenght"),
                    parameters.get<FPTYPE>("simulation.single_support_lenght")),

    mpc_model_(parameters.get<int>("simulation.N"),
               parameters.get<FPTYPE>("simulation.T"),
               robot_physical_parameters_->com_height())

{
    try {
        if(parameters.get<std::string>("simulation.logPredictions")=="true"){
            logPredictions_ = true;
        }
    } catch (boost::property_tree::ptree_bad_path e) {
            // Parameter has not been found. By default we consider this as false.
    }
}

// Constructor to use with python.
FormulationBase::FormulationBase(const std::string nameFileParameters): 
    InitParameters(nameFileParameters),
    current_iteration_(0),
    N_(parameters_.get<int>("simulation.N")),
    m_(parameters_.get<int>("simulation.m")),
    logPredictions_(false),
    robot_physical_parameters_(InstantiateRobotModel(parameters_.get<std::string>("simulation.robot"))),

    support_foot_position_(parameters_.get<FPTYPE>("initial_values.foot_x_position"),
                           parameters_.get<FPTYPE>("initial_values.foot_y_position"),
                           -robot_physical_parameters_->com_height()),

    dynamic_model_(parameters_.get<FPTYPE>("simulation.T"), robot_physical_parameters_->com_height(),
                   parameters_.get<FPTYPE>("initial_values.foot_x_position"), 
                   parameters_.get<FPTYPE>("initial_values.foot_y_position"),
                   parameters_.get<FPTYPE>("initial_values.foot_x_position"), 0.0, 0.0,
                   parameters_.get<FPTYPE>("initial_values.foot_y_position"), 0.0, 0.0),

    robot_restrictions_(robot_physical_parameters_.get(),
                        parameters_.get<int>("simulation.N"),
                        parameters_.get<int>("simulation.m")),

    step_generator_(parameters_.get<int>("simulation.N"),
                    parameters_.get<int>("simulation.m"),
                    Foot::RIGHT_FOOT,
                    parameters_.get<FPTYPE>("simulation.double_support_lenght"),
                    parameters_.get<FPTYPE>("simulation.single_support_lenght")),

    mpc_model_(parameters_.get<int>("simulation.N"),
               parameters_.get<FPTYPE>("simulation.T"),
               robot_physical_parameters_->com_height())

{
    try {
        if(parameters_.get<std::string>("simulation.logPredictions")=="true"){
            logPredictions_ = true;
        }
    } catch (boost::property_tree::ptree_bad_path e) {
            // Parameter has not been found. By default we consider this as false.
    }

    std::cout << "parameters_.get<int>(\"simulation.N\") = " << parameters_.get<int>("simulation.N") << std::endl;

}



Matrix_t FormulationBase::GetA()
{
    Matrix_t Pzu = mpc_model_.get_Pzu();
    Matrix_t U_future = step_generator_.GetFuture_U_Matrix();

    const Matrix_t &trunk_orientation = rotation_controller_->GetTrunkOrientation();
    const Matrix_t &feet_orientation  = rotation_controller_->GetFeetOrientation();

    Matrix_t feet = robot_restrictions_.GetFeetRestrictionsMatrix(feet_orientation);

    // For ZMP restrictions.
    Matrix_t Uc = step_generator_.GetCurrent_U_Matrix();
    Matrix_t Uf = step_generator_.GetFuture_U_Matrix();
    unsigned short int s1 = step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
    unsigned short int s2 = step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));
    Matrix_t SmVector(m_+1,1); SmVector.setZero();
    SmVector(0,0) = s1; SmVector(1,0) = s2; SmVector(2,0) = N_;
    Matrix_t thetas_support(m_+1,1); thetas_support.setZero();
    thetas_support(0,0) = rotation_controller_->GetSupportFootOrientation();
    thetas_support(1,0) = feet_orientation(s1-1,0);
    thetas_support(2,0) = feet_orientation(s2-1,0);
    Matrix_t zmp = robot_restrictions_.GetZMPRestrictionsMatrix(thetas_support, Pzu, U_future,SmVector);

    Matrix_t A(feet.rows() + zmp.rows(), feet.cols());
    A.setZero();
    A.block(0, 0, feet.rows(), feet.cols()) = feet;
    A.block(feet.rows(), 0, zmp.rows(), zmp.cols()) = zmp;

    return A;
}


Matrix_t FormulationBase::GetlbA()
{
    const Matrix_t &trunk_orientation = rotation_controller_->GetTrunkOrientation();
    const Matrix_t &feet_orientation  = rotation_controller_->GetFeetOrientation();

    Matrix_t Pzs = mpc_model_.get_Pzs();
    Matrix_t U_current = step_generator_.GetCurrent_U_Matrix();
    Matrix_t x_state = dynamic_model_.GetXStateVector();
    Matrix_t y_state = dynamic_model_.GetYStateVector();

    auto feet = robot_restrictions_.GetFeetLowerBoundaryVector(feet_orientation(0,0), support_foot_position_, step_generator_.GetSupportPhase(), step_generator_.GetFlyingFoot());

    // For ZMP restrictions.
    Matrix_t Uc = step_generator_.GetCurrent_U_Matrix();
    Matrix_t Uf = step_generator_.GetFuture_U_Matrix();
    unsigned short int s1 = step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
    unsigned short int s2 = step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));
    Matrix_t SmVector(m_+1,1); SmVector.setZero();
    SmVector(0,0) = s1; SmVector(1,0) = s2; SmVector(2,0) = N_;
    Matrix_t thetas_support(m_+1,1); thetas_support.setZero();
    thetas_support(0,0) = rotation_controller_->GetSupportFootOrientation();
    thetas_support(1,0) = feet_orientation(s1-1,0);
    thetas_support(2,0) = feet_orientation(s2-1,0);
    auto zmp = robot_restrictions_.GetZMPLowerBoundaryVector(support_foot_position_, thetas_support, Pzs, step_generator_, x_state, y_state,SmVector);

    Matrix_t lbA(feet.rows() + zmp.rows(), 1);
    lbA.block(0, 0, feet.rows(), feet.cols()) = feet;
    lbA.block(feet.rows(), 0, zmp.rows(), zmp.cols()) = zmp;

    return lbA;
}



Matrix_t FormulationBase::GetubA()
{
    const Matrix_t &trunk_orientation = rotation_controller_->GetTrunkOrientation();
    const Matrix_t &feet_orientation  = rotation_controller_->GetFeetOrientation();

    Matrix_t Pzs = mpc_model_.get_Pzs();
    Matrix_t U_current = step_generator_.GetCurrent_U_Matrix();
    Matrix_t x_state = dynamic_model_.GetXStateVector();
    Matrix_t y_state = dynamic_model_.GetYStateVector();

    auto feet = robot_restrictions_.GetFeetUpperBoundaryVector(feet_orientation(0,0), support_foot_position_, step_generator_.GetSupportPhase(), step_generator_.GetFlyingFoot());

    // For ZMP restrictions.
    Matrix_t Uc = step_generator_.GetCurrent_U_Matrix();
    Matrix_t Uf = step_generator_.GetFuture_U_Matrix();
    unsigned short int s1 = step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
    unsigned short int s2 = step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));
    Matrix_t SmVector(m_+1,1); SmVector.setZero();
    SmVector(0,0) = s1; SmVector(1,0) = s2; SmVector(2,0) = N_;
    Matrix_t thetas_support(m_+1,1); thetas_support.setZero();
    thetas_support(0,0) = rotation_controller_->GetSupportFootOrientation();
    thetas_support(1,0) = feet_orientation(s1-1,0);
    thetas_support(2,0) = feet_orientation(s2-1,0);
    auto zmp = robot_restrictions_.GetZMPUpperBoundaryVector(support_foot_position_, thetas_support, Pzs, step_generator_, x_state, y_state,SmVector);

    Matrix_t ubA(feet.rows() + zmp.rows(), 1);
    ubA.block(0, 0, feet.rows(), feet.cols()) = feet;
    ubA.block(feet.rows(), 0, zmp.rows(), zmp.cols()) = zmp;

    return ubA;
}

Matrix_t FormulationBase::Getlb()
{
    // no constraints, return empty vector
    Matrix_t lb;
    return lb;
}

Matrix_t FormulationBase::Getub()
{
    // no constraints, return empty vector
    Matrix_t lb;
    return lb;
}

int FormulationBase::GetNumberOfVariables()
{
    return N_ + m_ + N_ + m_;
}

int FormulationBase::GetNumberOfConstraints()
{
    return N_ + m_ + N_ + m_;
}