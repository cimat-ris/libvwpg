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

#ifndef FEET_H
#define FEET_H

#include <vector>

#include <Eigen/Dense>

#include "common.h"
#include "models.h"


enum class Foot {
    RIGHT_FOOT = 0,
    LEFT_FOOT = 1
};

enum class SupportPhase {
    DOUBLE_SUPPORT,
    SINGLE_SUPPORT
};


/******************************************************************************
 *
 *****************************************************************************/

class StepGenerator {

    public:
        StepGenerator(int N, int m, Foot support_foot, int dsl, int ssl);
        void UpdateSequence();
        inline bool IsSameSupportFoot() const {return is_same_support_foot_;};
        Matrix_t GetCurrent_U_Matrix() const;
        Matrix_t GetFuture_U_Matrix() const;
        inline Foot GetSupportFoot() const {return support_foot_;};
        inline Foot GetFlyingFoot() const {return static_cast<Foot>(1 - static_cast<int>(support_foot_));};
        inline SupportPhase GetSupportPhase() const {return support_phase_;};
        inline unsigned int GetRemainingSteps() const {return remaining_steps_;};
        inline int GetDoubleSupportLen() const {return double_support_len_;};
        inline int GetSingleSupportLen() const {return single_support_len_;};
        unsigned short int GetSelectionIndex( const Matrix_t &UFuture);
        Matrix_t GetDiagUMatrix(const Matrix_t &U);

    private:
        void ToggleSupportFoot();

    private:
        Foot support_foot_;
        SupportPhase support_phase_;
        unsigned int total_steps_;
        unsigned int remaining_steps_;
        const int N_;
        const int m_;
        const int double_support_len_;
        const int single_support_len_;
        bool is_same_support_foot_;
        Matrix_t steps_;
};


/******************************************************************************
 *
 *****************************************************************************/

class RobotRestrictions {

    public:

        RobotRestrictions(RobotModelInterface* robot_model, const int N, const int m);

        Matrix_t GetFeetRestrictionsMatrix(const Matrix_t& feet_angles);

        Matrix_t GetFeetLowerBoundaryVector(
                const FPTYPE initial_angle,
                const Point3D_t& foot_position,
                const SupportPhase& support_phase,
                const Foot& moving_foot);

        Matrix_t GetFeetUpperBoundaryVector(
                const FPTYPE initial_angle,
                const Point3D_t& foot_position,
                const SupportPhase& support_phase,
                const Foot& moving_foot);

        Matrix_t GetZMPRestrictionsMatrix(
                const Matrix_t &feet_angles,
                const Matrix_t &Pzu,
                const Matrix_t &UFuture,
                const Matrix_t &SmVector);

        Matrix_t GetZMPLowerBoundaryVector(
                const Point3D_t& foot_position, 
                const Matrix_t &feet_angles,
                const Matrix_t &Pzs,
                const StepGenerator &step_generator, 
                const Matrix_t &x_state,
                const Matrix_t &y_state,
                const Matrix_t &SmVector);

        Matrix_t GetZMPUpperBoundaryVector(
                const Point3D_t& foot_position,
                const Matrix_t &feet_angles,
                const Matrix_t &Pzs,
                const StepGenerator &step_generator, 
                const Matrix_t &x_state,
                const Matrix_t &y_state,
                const Matrix_t &SmVector);

        Matrix_t GetZMPOrientationMatrix_(const Matrix_t &feet_angles, const Matrix_t & SmVector);

/*************************   Nolineal constrains  ******************************/
        Matrix_t GetDerZMP(
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
            const Matrix_t &SmVector);

        Matrix_t GetAZMPNolinealRestrictions(
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
            const Matrix_t &SmVector);

        Matrix_t GetZMPNolinealLowerBoundaryVector(
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
            const Matrix_t &SmVector);


        Matrix_t GetZMPNolinealUpperBoundaryVector(
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
            const Matrix_t &SmVector);

        ///Feet restrictions

        Matrix_t GetDerFeet(
            const Matrix_t &feet_angles,
            const Matrix_t &Ppu,
            const Matrix_t &Xf,
            const Matrix_t &Yf,
            const StepGenerator &step_generator);

        Matrix_t GetAFootNolinealRestrictions(
            const Matrix_t &feet_angles,
            const Matrix_t &Ppu,
            const Matrix_t &Xf,
            const Matrix_t &Yf,
            const StepGenerator &step_generator);

        Matrix_t GetFeetNolinealLowerBoundaryVector(
            const Matrix_t &feet_angles,
            const Point3D_t& foot_position,
            const SupportPhase& support_phase,
            const Foot& moving_foot,
            const Matrix_t &jerks);

        Matrix_t GetFeetNolinealUpperBoundaryVector(
            const Matrix_t &feet_angles,
            const Point3D_t& foot_position,
            const SupportPhase& support_phase,
            const Foot& moving_foot,
            const Matrix_t &jerks);

        ////Orientations restrictions

        Matrix_t GetAOrientation(const Matrix_t &Ppu,
                                 const Matrix_t &Ppuk);

        Matrix_t GetOrientationLowerBoundaryVector(
            const Matrix_t &Pps,
            const Matrix_t &trunk_state,
            const Matrix_t &thetav);

        Matrix_t GetOrientationUpperBoundaryVector(
            const Matrix_t &Pps,
            const Matrix_t &trunk_state,
            const Matrix_t &thetav);
/*************************   Nolineal constrains  ******************************/

    private:

        RobotModelInterface* robot_model_;
        const int m_;
        const int N_;
        const FPTYPE MAX_THETA_ = 4.5 * M_PI / 180.0;

    private:

        void UpdateBoundariesForDoubleSupport_(
                Matrix_t& boundaries,
                const Foot& moving_foot,
                const Point3D_t& foot_position);

        Matrix_t GetCommonFeetRestrictionsVector_(
                const FPTYPE intial_angles,
                const Point3D_t& foot_position);

        Matrix_t GetCommonBoundaryVector(
                const Point3D_t& foot_position,
                const Matrix_t &angles,
                const Matrix_t &Pzs,
                const Matrix_t &Ucurrent,
                const Matrix_t &x_state,
                const Matrix_t &y_state,
                const Matrix_t &SmVector);
};

#endif
