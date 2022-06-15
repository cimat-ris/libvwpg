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

#ifndef TRUNK_ORIENTATION_H 
#define TRUNK_ORIENTATION_H

#include <vector>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include "common.h"
#include "qp.h"
#include "feet.h"


class RotationControlInterface {

    public:
        virtual const Matrix_t& GetFeetOrientation() = 0;
        virtual const Matrix_t& GetTrunkOrientation() = 0;
        virtual void UpdateReference(FPTYPE reference_angle) = 0;
		virtual void ComputeRobotNextOrientation() = 0;
        virtual void SetSupportFootOrientation(FPTYPE support_orientation_angle) = 0;
        virtual FPTYPE GetSupportFootOrientation() = 0;
        virtual FPTYPE GetCoMAngularSpeed() = 0;
        virtual FPTYPE GetCoMAngularAcceleration() = 0;
};


class RobotTrunkOrientationModel : public QProblemInterface {

    public:
        RobotTrunkOrientationModel(int N, FPTYPE T, int m, FPTYPE alpha, FPTYPE beta, FPTYPE gamma);
        Matrix_t GetH() override;
        Matrix_t Getg() override;
        Matrix_t GetA() override;
        Matrix_t GetlbA() override;
        Matrix_t GetubA() override;
        Matrix_t Getlb() override;
        Matrix_t Getub() override;
        int GetNumberOfVariables()  override;
        int GetNumberOfConstraints()  override;
         ~RobotTrunkOrientationModel() override;

        void SetReference(const FPTYPE reference_angle);
        Matrix_t RefAnglesInterpolation(const FPTYPE ref_angle);

        // Get
        const Matrix_t &GetPpu_();
	    const Matrix_t &GetPps_();
	    const Matrix_t &GetA_();
	    const Matrix_t &GetB_();
        const Vector3D_t &GetFootState_();
        const Vector3D_t &GetTrunkState_();

		//Set
		void SetFootState(const Vector3D_t &foot_state_new);
		void SetTrunkState(const Vector3D_t &trunk_state_new);
		void SetTrunkAngles(const Matrix_t &degresCoM);
		void SetFootAngles(const Matrix_t &degresFeet);

		//Get
		const Matrix_t &GetTrunkAngles();
		const Matrix_t &GetFootAngles();
        void PrintFootAngles();
        void PrintTrunkAngles();

        //For the new dynamical model of the flying foot
		const Matrix_t &GetPpuk_();
		void SetPpuk_();
		const Matrix_t &GetThetaV_();
		void SetThetaV_();
		Matrix_t RefAnglesFootInterpolation(const FPTYPE ref_angle);
        StepGenerator orientation_step_generator_;
        void SetSupportFootOrientation_(const FPTYPE angle);
        FPTYPE GetSupportFootOrientation_();

     private:
        void InitA(const FPTYPE T);
        void InitB(const FPTYPE T);
        void InitPps(const FPTYPE T);
        void InitPpu(const FPTYPE T);
        //For the flying foot.
		void InitPpuk();
		void InitThetaV();
        
    private:
        int N_;
        // const FPTYPE alpha_ = 0.05;
        // const FPTYPE beta_  = 100;
        // const FPTYPE gamma_ = 100;
        FPTYPE alpha_;
        FPTYPE beta_;
        FPTYPE gamma_;

        const FPTYPE MAX_THETA_ = 4.5 * M_PI / 180.0;
        const FPTYPE MAX_ORIENTATION_ = 2.0 * M_PI / 180.0;
        FPTYPE reference_angle_;

        Matrix_t Pps_;
        Matrix_t Ppu_;
        Matrix_t A_;
        Matrix_t B_;
        Matrix_t Ppuk_;
        Matrix_t thetaV_;
        Vector3D_t foot_state_;
        Vector3D_t trunk_state_;
		Matrix_t trunk_angles_;
		Matrix_t foot_angles_;
		FPTYPE support_foot_orientation_;
};


class RobotTrunkOrientationOptimizer : public RotationControlInterface {

    public:
        RobotTrunkOrientationOptimizer(int N, FPTYPE T, int m, FPTYPE alpha, FPTYPE beta, FPTYPE gamma);
        ~RobotTrunkOrientationOptimizer();

        void ComputeRobotNextOrientation() override;
        void SetSupportFootOrientation(FPTYPE support_orientation_angle) override ;
		FPTYPE GetSupportFootOrientation() override;
        FPTYPE GetCoMAngularSpeed() override;
		FPTYPE GetCoMAngularAcceleration() override;

        virtual const Matrix_t& GetFeetOrientation() override;
        virtual const Matrix_t& GetTrunkOrientation() override;
        void UpdateReference(FPTYPE reference_angle) override;

    private:
        int N_;
        FPTYPE target_angle_;
        RobotTrunkOrientationModel trunk_orientation_model_;
        QPSolver qp_solver_;
        void PrepareDataForNextOptimization(const Matrix_t &jerks);
};


class RobotTrunkOrientationFollower : public RotationControlInterface {
    public:
        RobotTrunkOrientationFollower(const std::vector<FPTYPE> &angles, const int N);
        ~RobotTrunkOrientationFollower();

        const Matrix_t& GetFeetOrientation() override;
        const Matrix_t& GetTrunkOrientation() override;
        void UpdateReference(FPTYPE reference_angle) override;
		void ComputeRobotNextOrientation() override ;
        void SetSupportFootOrientation(FPTYPE support_orientation_angle) override ;
		FPTYPE GetSupportFootOrientation() override;
        FPTYPE GetCoMAngularSpeed() override;
		FPTYPE GetCoMAngularAcceleration() override;

private:
        int N_;
        int current_iteration_;
        Matrix_t angles_;
        Matrix_t trunk_angles_;
};

#endif
