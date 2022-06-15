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

#ifndef QPSIMULATORS_H
#define QPSIMULATORS_H


#include <iostream>
#include <exception>
#include <string>
#include <map>

#include <Eigen/Dense>
#include "common.h"
#include "formulations.h"
#include <boost/lexical_cast.hpp>
#include "qp.h"
#include "feet.h"

//******************************************************************************
//
//                        QPHomographySimLinear
//
//******************************************************************************

class QPHomographySimLinear
{
    public:
        QPHomographySimLinear(const std::string nameFileParameters);
        FPTYPE GetCoMAbsoluteX();
        FPTYPE GetCoMAbsoluteY();
        FPTYPE GetCoMAbsoluteZ();
        FPTYPE GetCoMAbsoluteTheta();
        FPTYPE GetSupportFootAbsoluteX();
        FPTYPE GetSupportFootAbsoluteY();
        FPTYPE GetSupportFootAbsoluteZ();
        FPTYPE GetSupportFootAbsoluteTheta();
        FPTYPE GetNextSupportFootAbsoluteX();
        FPTYPE GetNextSupportFootAbsoluteY();
        FPTYPE GetNextSupportFootAbsoluteTheta();
        void SetCoMValues(const FPTYPE x_position, const FPTYPE y_position, const FPTYPE z_position, const FPTYPE x_speed, const FPTYPE y_speed );

        void SolveProblem();
        bool Continue();
        void Update();
        void LogCurrentResults();
        void GetObjectiveFunctionValue();

    protected:
        HomographySimulated simulator_;
        QPSolver qp_solver_;
};

//******************************************************************************
//
//                        QPHomographyRealLinear
//
//******************************************************************************
class QPHomographyRealLinear
{
    public:
        QPHomographyRealLinear(const std::string nameFileParameters, const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                               const FPTYPE h21, const FPTYPE h22, const FPTYPE h23,
                               const FPTYPE h31, const FPTYPE h32, const FPTYPE h33);
        FPTYPE GetCoMAbsoluteX();
        FPTYPE GetCoMAbsoluteY();
        FPTYPE GetCoMAbsoluteZ();
        FPTYPE GetCoMAbsoluteTheta();
        FPTYPE GetSupportFootAbsoluteX();
        FPTYPE GetSupportFootAbsoluteY();
        FPTYPE GetSupportFootAbsoluteZ();
        FPTYPE GetSupportFootAbsoluteTheta();
        FPTYPE GetNextSupportFootAbsoluteX();
        FPTYPE GetNextSupportFootAbsoluteY();
        FPTYPE GetNextSupportFootAbsoluteTheta();
        void SetCoMValues(const FPTYPE x_position, const FPTYPE y_position, const FPTYPE z_position, const FPTYPE x_speed, const FPTYPE y_speed );

        void SolveProblem();
        bool Continue();
        void Update();
        void LogCurrentResults();
        void GetObjectiveFunctionValue();
        void SetCurrentHomography(const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                                  const FPTYPE h21, const FPTYPE h22, const FPTYPE h23,
                                  const FPTYPE h31, const FPTYPE h32, const FPTYPE h33, 
                                  bool isComputeHomography);

    protected:
        Homography simulator_;
        QPSolver qp_solver_;
};

//******************************************************************************
//
//                        QPHomographyObstaclesLinear
//
//******************************************************************************

class QPHomographyObstaclesLinear
{
    public:
        QPHomographyObstaclesLinear(const std::string nameFileParameters, const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                                    const FPTYPE h21, const FPTYPE h23, const FPTYPE h31, const FPTYPE h32, const FPTYPE h33, 
                                    const FPTYPE c1,  const FPTYPE c2,  const FPTYPE c3, const bool isObstacle);
        FPTYPE GetCoMAbsoluteX();
        FPTYPE GetCoMAbsoluteY();
        FPTYPE GetCoMAbsoluteZ();
        FPTYPE GetCoMAbsoluteTheta();
        FPTYPE GetSupportFootAbsoluteX();
        FPTYPE GetSupportFootAbsoluteY();
        FPTYPE GetSupportFootAbsoluteZ();
        FPTYPE GetSupportFootAbsoluteTheta();
        FPTYPE GetNextSupportFootAbsoluteX();
        FPTYPE GetNextSupportFootAbsoluteY();
        FPTYPE GetNextSupportFootAbsoluteTheta();
        void SetCoMValues(const FPTYPE x_position, const FPTYPE y_position, const FPTYPE z_position, const FPTYPE x_speed, const FPTYPE y_speed );

        void SolveProblem();
        bool Continue();
        void Update();
        void LogCurrentResults();
        void GetObjectiveFunctionValue();
        void SetCurrentHomography( const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                                   const FPTYPE h21, const FPTYPE h23,
                                   const FPTYPE h31, const FPTYPE h32, const FPTYPE h33, const bool isObstacle, const bool isComputeHomography,
                                   const FPTYPE c1,  const FPTYPE c2,  const FPTYPE c3, const int iteration);
        bool IsSupportFootChange() ;
    protected:
        HomographyObstacles simulator_;
        QPSolver qp_solver_;
    
};

//******************************************************************************
//
//                        QPHerdtSim
//
//******************************************************************************

class QPHerdtSim
{
    public:
        QPHerdtSim(const std::string nameFileParameters);

        FPTYPE GetCoMAbsoluteX();
        FPTYPE GetCoMAbsoluteY();
        FPTYPE GetCoMAbsoluteZ();
        FPTYPE GetCoMAbsoluteTheta();
        FPTYPE GetSupportFootAbsoluteX();
        FPTYPE GetSupportFootAbsoluteY();
        FPTYPE GetSupportFootAbsoluteZ();
        FPTYPE GetSupportFootAbsoluteTheta();
        FPTYPE GetNextSupportFootAbsoluteX();
        FPTYPE GetNextSupportFootAbsoluteY();
        FPTYPE GetNextSupportFootAbsoluteTheta();
        void SetCoMValues(const FPTYPE x_position, const FPTYPE y_position, const FPTYPE z_position, const FPTYPE x_speed, const FPTYPE y_speed );

        void SolveProblem();
        bool Continue();
        void Update();
        void LogCurrentResults();
        void GetObjectiveFunctionValue();

    protected:
        Herdt simulator_;
        QPSolver qp_solver_;
    
};

//******************************************************************************
//
//                        QPHerdtReal
//
//******************************************************************************

class QPHerdtReal
{
    public:
        QPHerdtReal(const std::string nameFileParameters, const FPTYPE x_speed_ref, const FPTYPE y_speed_ref, const FPTYPE theta_ref );

        void SetCurrentReferenceSpeed(const FPTYPE x_speed_ref, const FPTYPE y_speed_ref, const FPTYPE theta_ref);

        FPTYPE GetCoMAbsoluteX();
        FPTYPE GetCoMAbsoluteY();
        FPTYPE GetCoMAbsoluteZ();
        FPTYPE GetCoMAbsoluteTheta();
        FPTYPE GetSupportFootAbsoluteX();
        FPTYPE GetSupportFootAbsoluteY();
        FPTYPE GetSupportFootAbsoluteZ();
        FPTYPE GetSupportFootAbsoluteTheta();
        FPTYPE GetNextSupportFootAbsoluteX();
        FPTYPE GetNextSupportFootAbsoluteY();
        FPTYPE GetNextSupportFootAbsoluteTheta();
        void SetCoMValues(const FPTYPE x_position, const FPTYPE y_position, const FPTYPE z_position, const FPTYPE x_speed, const FPTYPE y_speed );

        void SolveProblem();
        bool Continue();
        void Update();
        void LogCurrentResults();
        void GetObjectiveFunctionValue();

    protected:
        HerdtReal simulator_;
        QPSolver qp_solver_;

};

#endif //QPSIMULATORS_H
