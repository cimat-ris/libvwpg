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

#ifndef COMMON_FORMULATION_H
#define COMMON_FORMULATION_H

#include <vector>
#include <memory>

#include <boost/property_tree/ptree.hpp>

#include "feet.h"
#include "common.h"
#include "models.h"
#include "qp.h"
#include "trunk_orientation.h"


class SimulatorInterface {
    public:
        virtual void Update(const Matrix_t &solution) = 0;
        virtual void LogCurrentResults(const Matrix_t &solution=Matrix_t()) const = 0;
        virtual void LogCurrentPredictions(const Matrix_t &solution=Matrix_t()) const = 0;     
        virtual Point3D_t GetCoMAbsolutePosition() const = 0;   
        virtual FPTYPE GetCoMAbsoluteTheta() const = 0;   
        virtual FPTYPE GetCurrentStepRatio() const = 0; 
        virtual int GetCurrentStepRemainingCycles() const = 0;     
        virtual bool GetCurrentSupportType() const = 0;     
        virtual bool GetSupportFootId() const = 0;
        virtual Point3D_t GetSupportFootAbsolutePosition() const = 0;   
        virtual FPTYPE GetSupportFootAbsoluteTheta() const = 0;      
        virtual FPTYPE GetSwingFootAbsoluteTheta() const = 0; 
        virtual Point3D_t GetNextSupportFootAbsolutePosition(const Matrix_t &solution=Matrix_t()) const = 0;   
        virtual FPTYPE GetNextSupportFootAbsoluteTheta() const = 0; 
        virtual void SetCoMValues(const FPTYPE x_position, const FPTYPE y_position, const FPTYPE z_position, const FPTYPE x_speed, const FPTYPE y_speed ) = 0; 
        virtual bool IsSupportFootChange() const = 0;         
        virtual bool Continue() const = 0;
        virtual ~SimulatorInterface() { };
};

//To use with python
class InitParameters
{
    public:
        InitParameters(const std::string nameFileParameters);
        InitParameters(const boost::property_tree::ptree &parameters);
        virtual ~InitParameters(){};
    protected:
        boost::property_tree::ptree parameters_;
        
};


class FormulationBase : public QProblemInterface, public InitParameters {
    public:
        FormulationBase(const boost::property_tree::ptree &parameters);
        FormulationBase(const std::string nameFileParameters);
        virtual ~FormulationBase() {}

        Matrix_t GetH() override = 0;
        Matrix_t Getg() override = 0;
        Matrix_t GetA() override;
        Matrix_t GetlbA() override;
        Matrix_t GetubA() override;
        Matrix_t Getlb() override;
        Matrix_t Getub() override;
        int GetNumberOfVariables() override;
        int GetNumberOfConstraints() override;
        inline RobotModelInterface *getRobotPhysicalParameters() { return robot_physical_parameters_.get();};

    protected:
        boost::shared_ptr<RobotModelInterface> robot_physical_parameters_;
        MPCModel mpc_model_;
        RobotDynamicModel dynamic_model_;
        RobotRestrictions robot_restrictions_;
        StepGenerator step_generator_;

        int N_;
        int m_;
        bool logPredictions_;
        int current_iteration_;
        Point3D_t support_foot_position_;
        boost::shared_ptr<RotationControlInterface> rotation_controller_;
        
};

#endif
