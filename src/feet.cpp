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
#include <iostream>
#include <exception>

#include <Eigen/Dense>

#include "feet.h"
#include "common.h"
#include "models.h"

using namespace Eigen;

namespace {

    inline Matrix_t XkDependentOnX(const FPTYPE angle)
    {
        Matrix_t tmp(1, 2);
        tmp << -std::cos(angle) , std::cos(angle);
        return tmp;
    }


    inline Matrix_t XkDependentOnY(const FPTYPE angle)
    {
        Matrix_t tmp(1, 2);
        tmp << -std::sin(angle) , std::sin(angle);
        return tmp;
    }

    inline Matrix_t YkDependentOnX(const FPTYPE angle)
    {
        Matrix_t tmp(1, 2);
        tmp << std::sin(angle) , -std::sin(angle);
        return tmp;
    }

    inline Matrix_t YkDependentOnY(const FPTYPE angle)
    {
        Matrix_t tmp(1, 2);
        tmp << -std::cos(angle) , std::cos(angle);
        return tmp;
    }

}

/******************************************************************************
 *
 *                              RobotRestrictions
 *
 ******************************************************************************/

// public:

RobotRestrictions::RobotRestrictions(RobotModelInterface* robot_model, const int N, const int m):
    robot_model_(robot_model), N_(N), m_(m)
{

}

Matrix_t RobotRestrictions::GetFeetRestrictionsMatrix(const Matrix_t& feet_angles)
{

    Matrix_t foot_restrictions(2*m_, 2*N_ + 2*m_);
    foot_restrictions.setZero();

    foot_restrictions(0, N_) = std::cos(feet_angles(0,0));
    foot_restrictions(0, N_+m_+N_) = std::sin(feet_angles(0,0));

    foot_restrictions(m_, N_) = -std::sin(feet_angles(0,0));
    foot_restrictions(m_, N_+m_+N_) = std::cos(feet_angles(0,0));


    int column = 0;
    for (int i=1; i<m_; i++) {
        foot_restrictions.block(i, N_+column, 1, 2) = XkDependentOnX(feet_angles(i,0));
        foot_restrictions.block(i, N_ + m_ + N_ + column, 1, 2) =  XkDependentOnY(feet_angles(i,0));
        ++column;
    }

    column = 0;
    for (int i=m_+1; i<m_+m_; i++) {
        foot_restrictions.block(i, N_+column, 1, 2) = YkDependentOnX(feet_angles(i-m_,0));
        foot_restrictions.block(i, N_ + m_ + N_ + column, 1, 2) = YkDependentOnY(feet_angles(i-m_,0));
        ++column;
    }

    return  foot_restrictions;
}

Matrix_t RobotRestrictions::GetFeetLowerBoundaryVector(
        const FPTYPE initial_angle,
        const Point3D_t& foot_position,
        const SupportPhase& support_phase,
        const Foot& moving_foot)
{
    Matrix_t boundaries = GetCommonFeetRestrictionsVector_(initial_angle, foot_position);

    bool update_left_foot = false;
    if (moving_foot == Foot::LEFT_FOOT)
        update_left_foot = true;

    // x constraints
    for (int i=0; i<m_; i++)
        boundaries(i, 0) -= robot_model_->max_step_len();

    // y constraints
    for (int i=m_; i<2*m_; i++) {
        if (update_left_foot)
            boundaries(i, 0) += robot_model_->min_feet_dist();
        else
            boundaries(i, 0) -= robot_model_->max_feet_dist();

        update_left_foot = !update_left_foot;
    }

    if (support_phase == SupportPhase::DOUBLE_SUPPORT)
        UpdateBoundariesForDoubleSupport_(boundaries, moving_foot, foot_position);

    return boundaries;
}

Matrix_t RobotRestrictions::GetFeetUpperBoundaryVector(
        const FPTYPE initial_angle,
        const Point3D_t& foot_position,
        const SupportPhase& support_phase,
        const Foot& moving_foot)
{
    Matrix_t boundaries = GetCommonFeetRestrictionsVector_(initial_angle, foot_position);

    bool update_left_foot = false;
    if (moving_foot == Foot::LEFT_FOOT)
        update_left_foot = true;

    // x constraints
    for (int i=0; i<m_; i++)
        boundaries(i, 0) += robot_model_->max_step_len();

    // y constraints
    for (int i=m_; i<2*m_; i++) {
        if (update_left_foot)
            boundaries(i, 0) += robot_model_->max_feet_dist();
        else
            boundaries(i, 0) -= robot_model_->min_feet_dist();

        update_left_foot = !update_left_foot;
    }

    if (support_phase == SupportPhase::DOUBLE_SUPPORT)
        UpdateBoundariesForDoubleSupport_(boundaries, moving_foot, foot_position);

    return boundaries;
}

Matrix_t RobotRestrictions::GetZMPRestrictionsMatrix(
        const Matrix_t &feet_angles,
        const Matrix_t &Pzu,
        const Matrix_t &UFuture,
        const Matrix_t &SmVector)
{
    Matrix_t restrictions_matrix(2*Pzu.rows() , 2*Pzu.cols() + 2*UFuture.cols());
    restrictions_matrix.setZero();
    // See DAUxy in Noe's code
    restrictions_matrix.block(0, 0, Pzu.rows(), Pzu.cols()) = Pzu;
    restrictions_matrix.block(0, Pzu.cols(), UFuture.rows(), UFuture.cols()) = -UFuture;
    restrictions_matrix.block(Pzu.rows(), Pzu.cols() + UFuture.cols(), Pzu.rows(), Pzu.cols()) = Pzu;
    restrictions_matrix.block(Pzu.rows(), Pzu.cols() + UFuture.cols() + Pzu.cols(), UFuture.rows(), UFuture.cols()) = -UFuture;

    return GetZMPOrientationMatrix_(feet_angles,SmVector)*restrictions_matrix;
}

Matrix_t RobotRestrictions::GetCommonBoundaryVector(
        const Point3D_t& foot_position,
        const Matrix_t &feet_angles,
        const Matrix_t &Pzs,
        const Matrix_t &Ucurrent,
        const Matrix_t &x_state,
        const Matrix_t &y_state,
        const Matrix_t &SmVector)
{
    Matrix_t boundaries(2*N_, 1);
    boundaries.setZero();
    boundaries.block(0, 0, N_, 1)  = Ucurrent * X(foot_position) - Pzs * x_state;
    boundaries.block(N_, 0, N_, 1) = Ucurrent * Y(foot_position) - Pzs * y_state;
    return GetZMPOrientationMatrix_(feet_angles,SmVector)*boundaries;
}

Matrix_t RobotRestrictions::GetZMPLowerBoundaryVector(
        const Point3D_t& foot_position,
        const Matrix_t &feet_angles,
        const Matrix_t &Pzs,
        const StepGenerator &step_generator,
        const Matrix_t &x_state,
        const Matrix_t &y_state,
        const Matrix_t &SmVector)
{
    const Matrix_t Ucurrent = step_generator.GetCurrent_U_Matrix();
    Matrix_t boundaries =
        GetCommonBoundaryVector(foot_position, feet_angles, Pzs, Ucurrent, x_state, y_state,SmVector);
                                                                                 
    for (int i = 0; i <= 2*N_ - 2; i += 2) {                                    
        boundaries(i + 0, 0) = -0.5 * robot_model_->foot_len() + boundaries(i + 0, 0);
        boundaries(i + 1, 0) = -0.5 * robot_model_->foot_width() + boundaries(i + 1, 0);                 
    }                                                                           

    if (step_generator.GetSupportPhase() == SupportPhase::DOUBLE_SUPPORT) {     
        int row = 0;                                                            
        for (int i=0; i < step_generator.GetRemainingSteps(); i++) {            
            boundaries(row + 0, 0) = -0.25 *(robot_model_->foot_width() + robot_model_->feet_dist_default());  
            boundaries(row + 1, 0) = -0.5 * robot_model_->foot_len();   
            row += 2;                                                           
        }                                                                       
    }  

    return boundaries;
}

Matrix_t RobotRestrictions::GetZMPUpperBoundaryVector(
        const Point3D_t& foot_position,
        const Matrix_t &feet_angles,
        const Matrix_t &Pzs,
        const StepGenerator &step_generator,
        const Matrix_t &x_state,
        const Matrix_t &y_state,
        const Matrix_t &SmVector)
{
    const Matrix_t Ucurrent = step_generator.GetCurrent_U_Matrix();
    Matrix_t boundaries =
        GetCommonBoundaryVector(foot_position, feet_angles, Pzs, Ucurrent, x_state, y_state,SmVector);

    for (int i = 0; i <= 2*N_ - 2; i += 2) {                                    
        boundaries(i + 0, 0) += 0.5 * robot_model_->foot_len();                   
        boundaries(i + 1, 0) += 0.5 * robot_model_->foot_width();                 
    }                                                                           

    if (step_generator.GetSupportPhase() == SupportPhase::DOUBLE_SUPPORT) {     
        int row = 0;                                                            
        for (int i=0; i<step_generator.GetRemainingSteps(); i++) {              
            boundaries(row + 0, 0) = 0.25 *(robot_model_->foot_width() + robot_model_->feet_dist_default());  
            boundaries(row + 1, 0) = 0.5 * robot_model_->foot_len();   
            row += 2;                                                           
        }                                                                       
    }    

    return boundaries;
}

// private:
Matrix_t RobotRestrictions::GetCommonFeetRestrictionsVector_(
    const FPTYPE initial_angle,
    const Point3D_t& foot_position)
{
    Matrix_t restrictions_vector(m_ + m_, 1);

    restrictions_vector.setZero();
    restrictions_vector(0, 0) = X(foot_position) * std::cos(initial_angle) + Y(foot_position) * std::sin(initial_angle);
    restrictions_vector(m_, 0) = -X(foot_position) * std::sin(initial_angle) + Y(foot_position) * std::cos(initial_angle);

    return restrictions_vector;
}

void RobotRestrictions::UpdateBoundariesForDoubleSupport_(
        Matrix_t& boundaries,
        const Foot& moving_foot,
        const Point3D_t& foot_position)
{
        boundaries(0, 0) = X(foot_position);

        if (moving_foot == Foot::LEFT_FOOT)
            boundaries(m_, 0) = Y(foot_position) + robot_model_->feet_dist_default() / 2;
        else
            boundaries(m_, 0) = Y(foot_position) - robot_model_->feet_dist_default() / 2;
}

Matrix_t RobotRestrictions::GetZMPOrientationMatrix_(const Matrix_t &feet_angles, const Matrix_t &SmVector)
{
    Matrix_t restrictions_matrix(2*N_, 2*N_);
    restrictions_matrix.setZero();

    int s0 = 0;
    int s;
    int row = 0;

    for(int i=0; i<SmVector.size(); i++)
    {
        // See DU0 in Noe's code
        s = SmVector(i,0);

        for (int j=s0; j<s; j++) {
            restrictions_matrix(row + 0, j) =  std::cos(feet_angles(i,0));
            restrictions_matrix(row + 1, j) = -std::sin(feet_angles(i,0));

            row += 2;
        }
        s0 = s;
    }
    row = 0;
    s0  = 0;
    for(int i=0; i<SmVector.size(); i++)
    {
        s = SmVector(i,0);
        for (int j=(N_+s0); j<(N_+s); j++) {
            restrictions_matrix(row + 0, j) = std::sin(feet_angles(i,0));
            restrictions_matrix(row + 1, j) = std::cos(feet_angles(i,0));
            row += 2;
        }
        s0 = s;
    }

    return restrictions_matrix;
}

/*************************************** No lineal restrictions **************************************/
/// ZMP nolinear restrictions
Matrix_t RobotRestrictions::GetDerZMP(
        const Matrix_t &feet_angles,
        const Matrix_t &Pzs,
        const Matrix_t &Pzu,
        const Matrix_t &Ppu,
        const Matrix_t &UFuture,
        const Matrix_t &Ucurrent,
        const Point3D_t &foot_position,
        const Matrix_t &jerks_x,
        const Matrix_t &jerks_y,
        const Matrix_t &x_state,
        const Matrix_t &y_state,
        const Matrix_t &Xf,
        const Matrix_t &Yf,
        const Matrix_t &SmVector )
{
    Matrix_t DAUf(2*N_,N_);
    DAUf.setZero();
    // See DAUt in Noe's code
    Matrix_t Fx = Pzu * jerks_x - UFuture * Xf ;
    Matrix_t Fy = Pzu * jerks_y - UFuture * Yf ;

    int s0;
    int s;

    for(int k=0;k<(SmVector.size()-1);k++)
    {
        s0 = SmVector(k,0);
        s  = SmVector(k+1,0);
        for(int i=s0;i<s; i++)
        {
            for(int j=0;j<N_; j++)
            {
                DAUf(2*i  ,j) = -sin(feet_angles(k+1,0))*Ppu(k+1,j)*Fx(i)  + cos(feet_angles(k+1,0))*Ppu(k+1,j)*Fy(i);
                DAUf(2*i+1,j) = -cos(feet_angles(k+1,0))*Ppu(k+1,j)*Fx(i)  - sin(feet_angles(k+1,0))*Ppu(k+1,j)*Fy(i);
            }
        }
    }

    return DAUf;
}

Matrix_t RobotRestrictions::GetAZMPNolinealRestrictions(
        const Matrix_t &feet_angles,
        const Matrix_t &Pzs,
        const Matrix_t &Pzu,
        const Matrix_t &Ppu,
        const Matrix_t &UFuture,
        const Matrix_t &Ucurrent,
        const Point3D_t &foot_position,
        const Matrix_t &jerks_x,
        const Matrix_t &jerks_y,
        const Matrix_t &x_state,
        const Matrix_t &y_state,
        const Matrix_t &Xf,
        const Matrix_t &Yf,
        const Matrix_t &SmVector)
{
    // For 2N_ <= or >= constraints (zmp positions)
    Matrix_t DAUxy = GetZMPRestrictionsMatrix(feet_angles,Pzu,UFuture,SmVector);
    Matrix_t DAUf  = GetDerZMP(feet_angles,Pzs,Pzu,Ppu,UFuture,Ucurrent,foot_position,jerks_x,jerks_y,x_state,y_state,Xf,Yf,SmVector);
    Matrix_t C1(2 * N_, 4 * N_ + 2 * m_);
    C1.setZero();
    C1.block(0,0,2 * N_,2 * N_ + 2*m_)    = DAUxy;
    C1.block(0,2 * N_ + 2 * m_,2 * N_,N_) = DAUf;
    return C1;
}

Matrix_t RobotRestrictions::GetZMPNolinealLowerBoundaryVector(
        const Matrix_t &feet_angles,
        const Matrix_t &Pzs,
        const Matrix_t &Pzu,
        const StepGenerator &step_generator,
        const Matrix_t &UFuture,
        const Matrix_t &Ucurrent,
        const Point3D_t &foot_position,
        const Matrix_t &jerks_x,
        const Matrix_t &jerks_y,
        const Matrix_t &x_state,
        const Matrix_t &y_state,
        const Matrix_t &Xf,
        const Matrix_t &Yf,
        const Matrix_t &SmVector)
{
    Matrix_t F(2 * N_,1);
    F.setZero();
    F.block(0,0,N_,1)  = Pzu * jerks_x - UFuture * Xf;
    F.block(N_,0,N_,1) = Pzu * jerks_y - UFuture * Yf;
    Matrix_t D  = GetZMPOrientationMatrix_(feet_angles,SmVector);
    return (GetZMPLowerBoundaryVector(foot_position,feet_angles,Pzs,step_generator,x_state,y_state,SmVector) - D * F );

}

Matrix_t RobotRestrictions::GetZMPNolinealUpperBoundaryVector(
        const Matrix_t &feet_angles,
        const Matrix_t &Pzs,
        const Matrix_t &Pzu,
        const StepGenerator &step_generator,
        const Matrix_t &UFuture,
        const Matrix_t &Ucurrent,
        const Point3D_t &foot_position,
        const Matrix_t &jerks_x,
        const Matrix_t &jerks_y,
        const Matrix_t &x_state,
        const Matrix_t &y_state,
        const Matrix_t &Xf,
        const Matrix_t &Yf,
        const Matrix_t &SmVector)
{
    Matrix_t F(2 * N_,1);
    F.setZero();
    F.block(0,0,N_,1)  = Pzu * jerks_x - UFuture * Xf;
    F.block(N_,0,N_,1) = Pzu * jerks_y - UFuture * Yf;
    Matrix_t D  = GetZMPOrientationMatrix_(feet_angles,SmVector);
    return ( GetZMPUpperBoundaryVector(foot_position,feet_angles,Pzs,step_generator,x_state,y_state,SmVector) - D * F );

}
/// Feet nolinear restrictions
Matrix_t RobotRestrictions::GetDerFeet(
        const Matrix_t &feet_angles,
        const Matrix_t &Ppu,
        const Matrix_t &Xf,
        const Matrix_t &Yf,
        const StepGenerator &step_generator
        )
{
    Matrix_t DAfUt(2 * m_, N_);
    DAfUt.setZero();
    FPTYPE Xrem;
    FPTYPE Yrem;

    unsigned int ind = 0;
    for(unsigned int j=0;j<N_;j++)
    {
        DAfUt(0,j) = -Xf(0) * sin(feet_angles(0,0)) * Ppu(ind,j) + Yf(0) * cos(feet_angles(0,0)) * Ppu(ind,j);
        DAfUt(1,j) = -Xf(0) * cos(feet_angles(0,0)) * Ppu(ind,j) - Yf(0) * sin(feet_angles(0,0)) * Ppu(ind,j);
    } 

    for(unsigned int i=1;i<m_;i++)
    {
        ind += step_generator.GetSingleSupportLen();
        Xrem = (-Xf(i-1) + Xf(i)); 
        Yrem = (-Yf(i-1) + Yf(i));
        for(unsigned int j=0;j<N_;j++)
        {
            DAfUt(2*i  ,j) = -Xrem * sin(feet_angles(ind,0)) * Ppu(ind,j) + Yrem * cos(feet_angles(ind,0)) * Ppu(ind,j);
            DAfUt(2*i+1,j) = -Xrem * cos(feet_angles(ind,0)) * Ppu(ind,j) - Yrem * sin(feet_angles(ind,0)) * Ppu(ind,j);
        }
    }

    return DAfUt;
}

Matrix_t RobotRestrictions::GetAFootNolinealRestrictions(
            const Matrix_t &feet_angles,
            const Matrix_t &Ppu,
            const Matrix_t &Xf,
            const Matrix_t &Yf,
            const StepGenerator &step_generator
            )
{
    // Size 2*m_x(2*N_ + 2*m_)
    Matrix_t DAUxy = GetFeetRestrictionsMatrix(feet_angles);
    Matrix_t DAfUt = GetDerFeet(feet_angles,Ppu,Xf,Yf,step_generator);

    Matrix_t C2(2 * m_, 4 * N_ + 2 * m_);
    C2.setZero();
    C2.block(0,0,2 * m_,2 * N_ + 2 * m_)    = DAUxy;
    C2.block(0,2 * N_ + 2 * m_,2 * m_,N_)   = DAfUt;
    return C2;
}

Matrix_t RobotRestrictions::GetFeetNolinealLowerBoundaryVector(
        const Matrix_t &foot_angles,
        const Point3D_t& foot_position,
        const SupportPhase& support_phase,
        const Foot& moving_foot,
        const Matrix_t &jerks)

{
    // Size 2*m_x(2*N_ + 2*m_)
    Matrix_t D = GetFeetRestrictionsMatrix(foot_angles);
    return (GetFeetLowerBoundaryVector(foot_angles(0,0),foot_position,support_phase,moving_foot) - D * jerks.block(0,0,2 * N_ + 2 * m_,1));
}


Matrix_t RobotRestrictions::GetFeetNolinealUpperBoundaryVector(
        const Matrix_t &foot_angles,
        const Point3D_t& foot_position,
        const SupportPhase& support_phase,
        const Foot& moving_foot,
        const Matrix_t &jerks)

{
    // Size 2*m_x(2*N_ + 2*m_)
    Matrix_t D = GetFeetRestrictionsMatrix(foot_angles);
    return (GetFeetUpperBoundaryVector(foot_angles(0,0),foot_position,support_phase,moving_foot) - D * jerks.block(0,0,2 * N_ + 2 * m_,1));    
}

///Orientation restrictions
Matrix_t RobotRestrictions::GetAOrientation(const Matrix_t &Ppu, const Matrix_t &Ppuk)
{
    Matrix_t C3(N_,4 * N_ + 2 * m_);
    C3.setZero();
    C3.block(0,2 * N_ + 2 * m_,N_,N_) = Ppuk; //Foot
    C3.block(0,3 * N_ + 2 * m_,N_,N_) = -Ppu; //trunk
    return C3;
}

Matrix_t RobotRestrictions::GetOrientationLowerBoundaryVector(
        const Matrix_t &Pps,
        const Matrix_t &trunk_state,
        const Matrix_t &thetaV)
{
    // TODO: in the nonlinear case, this function seems to suppose that U0=0?
    Matrix_t lbA(N_, 1);
    lbA.setZero();
    lbA = -MAX_THETA_ * Matrix_t::Ones(N_, 1) + (Pps * trunk_state) - thetaV;
    return lbA;
}

Matrix_t RobotRestrictions::GetOrientationUpperBoundaryVector(
        const Matrix_t &Pps,
        const Matrix_t &trunk_state,
        const Matrix_t &thetaV)
{
    // TODO: in the nonlinear case, this function seems to suppose that U0=0?
    Matrix_t ubA(N_, 1);
    ubA.setZero();
    ubA = MAX_THETA_ * Matrix_t::Ones(N_, 1) + (Pps * trunk_state) - thetaV;

    return ubA;
}
/*************************************** No lineal restrictions **************************************/


/******************************************************************************
 *
 *                               StepGenerator
 *
 ******************************************************************************/

// public:

StepGenerator::StepGenerator(int N, int m, Foot support_foot, int dsl=2, int ssl=7):
    N_(N),
    m_(m),
    support_foot_(support_foot),
    double_support_len_(dsl),
    single_support_len_(ssl),
    support_phase_(SupportPhase::DOUBLE_SUPPORT),
    steps_(N, m+1),
    total_steps_(0),
    is_same_support_foot_(true),
    remaining_steps_(dsl)
{
    if (N_-1 != m*single_support_len_) {
        std::cerr << "ERROR: Foot selection matrix could not be created, the following "
            << " condition must be met: N-1 == m*SINGLE_SUPPORT_LENGTH" << std::endl;
        throw std::exception();
    }

    steps_.setZero();

    for (int i=0; i<double_support_len_; i++)
        steps_(i,0) = 1;

    int row = double_support_len_;
    for (int col=1; col<m+1; col++) {
        for (int taken_steps = 0; row<N && taken_steps < single_support_len_; taken_steps++, row++)
            steps_(row, col) = 1;
    }
}

Matrix_t StepGenerator::GetCurrent_U_Matrix() const
{
    return steps_.block(0, 0, steps_.rows(), 1);
}

Matrix_t StepGenerator::GetFuture_U_Matrix() const
{
    return steps_.block(0, 1, steps_.rows(), steps_.cols()-1);
}

void StepGenerator::UpdateSequence()
{
    ++total_steps_;
    --remaining_steps_;
    is_same_support_foot_= true;

    // Check if first column would become zero after moving submatrix, move submatrix accordingly
    if (((int)steps_(1, 0)) == 0) { 
        // Copy block starting at 1,1 to 0,0
        steps_.block(0, 0, steps_.rows()-1, steps_.cols()-1) = steps_.block(1, 1, steps_.rows()-1, steps_.cols()-1);
        // Last column set to zero
        steps_.block(0, steps_.cols()-1, steps_.rows(), 1).setZero();
        ToggleSupportFoot();
    } else {
        steps_.block(0, 0, steps_.rows()-1, steps_.cols()) = steps_.block(1, 0, steps_.rows()-1, steps_.cols());
        steps_.block(steps_.rows()-1, 0, 1, steps_.cols()).setZero();
    }

    // Set element in the lower right corner to 1
    steps_(steps_.rows()-1, steps_.cols()-1) = 1;
}

// private:
void StepGenerator::ToggleSupportFoot()
{
    is_same_support_foot_ = false;
    // At the first change of support foot, pass from double support to single support
    support_phase_   = SupportPhase::SINGLE_SUPPORT;
    remaining_steps_ = single_support_len_;
    support_foot_ = static_cast<Foot>(1 - static_cast<int>(support_foot_));
}

// Recover the first non-null index in a selection column
unsigned short int StepGenerator::GetSelectionIndex(const Matrix_t &UFuture)
{
    for(unsigned short int i=0;i<N_;i++){
        if(UFuture(i,0) == 1)
            return i;
    }
    return 0;
}

Matrix_t StepGenerator::GetDiagUMatrix(const Matrix_t &U )
{
    Matrix_t diagU(N_,N_); diagU.setZero();
    for(int i=0; i<N_; i++)
        diagU(i,i) = U(i,0);
    return  diagU;
}