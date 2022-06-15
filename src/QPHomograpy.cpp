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

#include <boost/python.hpp>
#include <iostream>
#include <exception>
#include <string>
#include <map>

#include <Eigen/Dense>
#include "common.h"
#include "qpOASES.hpp"
#include "formulations.h"
#include "QPSimulators.h"


#include <boost/python/wrapper.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/exceptions.hpp>

using namespace boost::python;

BOOST_PYTHON_MODULE(QPHomograpy_py)
{

	class_<QPHomographySimLinear>("QPHomographySimLinear",init<std::string>())
		.def("GetCoMAbsoluteX",&QPHomographySimLinear::GetCoMAbsoluteX)
		.def("GetCoMAbsoluteY",&QPHomographySimLinear::GetCoMAbsoluteY)
		.def("GetCoMAbsoluteZ",&QPHomographySimLinear::GetCoMAbsoluteZ)
		.def("GetCoMAbsoluteTheta",&QPHomographySimLinear::GetCoMAbsoluteTheta)
		.def("GetSupportFootAbsoluteX",&QPHomographySimLinear::GetSupportFootAbsoluteX)
		.def("GetSupportFootAbsoluteY",&QPHomographySimLinear::GetSupportFootAbsoluteY)
		.def("GetSupportFootAbsoluteZ",&QPHomographySimLinear::GetSupportFootAbsoluteZ)
		.def("GetNextSupportFootAbsoluteX",&QPHomographySimLinear::GetNextSupportFootAbsoluteX)
		.def("GetNextSupportFootAbsoluteY",&QPHomographySimLinear::GetNextSupportFootAbsoluteY)
		.def("GetSupportFootAbsoluteTheta",&QPHomographySimLinear::GetSupportFootAbsoluteTheta)
		.def("GetNextSupportFootAbsoluteTheta",&QPHomographySimLinear::GetNextSupportFootAbsoluteTheta)
		.def("SolveProblem",&QPHomographySimLinear::SolveProblem)
		.def("Continue",&QPHomographySimLinear::Continue)
		.def("Update",&QPHomographySimLinear::Update)
		.def("LogCurrentResults",&QPHomographySimLinear::LogCurrentResults)
		.def("SetCoMValues",&QPHomographySimLinear::SetCoMValues,args("x_position","y_position", "z_position","x_speed", "y_speed"))
		.def("GetObjectiveFunctionValue",&QPHomographySimLinear::GetObjectiveFunctionValue)
		;

	class_<QPHomographyRealLinear>("QPHomographyRealLinear",init<std::string,FPTYPE,FPTYPE,FPTYPE,FPTYPE,FPTYPE,FPTYPE,FPTYPE,FPTYPE,FPTYPE >())
        .def("GetCoMAbsoluteX",&QPHomographyRealLinear::GetCoMAbsoluteX)
        .def("GetCoMAbsoluteY",&QPHomographyRealLinear::GetCoMAbsoluteY)
        .def("GetCoMAbsoluteZ",&QPHomographyRealLinear::GetCoMAbsoluteZ)
        .def("GetCoMAbsoluteTheta",&QPHomographyRealLinear::GetCoMAbsoluteTheta)
        .def("GetSupportFootAbsoluteX",&QPHomographyRealLinear::GetSupportFootAbsoluteX)
        .def("GetSupportFootAbsoluteY",&QPHomographyRealLinear::GetSupportFootAbsoluteY)
        .def("GetSupportFootAbsoluteZ",&QPHomographyRealLinear::GetSupportFootAbsoluteZ)
        .def("GetNextSupportFootAbsoluteX",&QPHomographyRealLinear::GetNextSupportFootAbsoluteX)
        .def("GetNextSupportFootAbsoluteY",&QPHomographyRealLinear::GetNextSupportFootAbsoluteY)
        .def("GetSupportFootAbsoluteTheta",&QPHomographyRealLinear::GetSupportFootAbsoluteTheta)
        .def("GetNextSupportFootAbsoluteTheta",&QPHomographyRealLinear::GetNextSupportFootAbsoluteTheta)
        .def("SolveProblem",&QPHomographyRealLinear::SolveProblem)
        .def("Continue",&QPHomographyRealLinear::Continue)
        .def("Update",&QPHomographyRealLinear::Update)
        .def("LogCurrentResults",&QPHomographyRealLinear::LogCurrentResults)
        .def("SetCurrentHomography",&QPHomographyRealLinear::SetCurrentHomography,args("h11","h12","h13","h21","h22","h23","h31","h32","h33","isComputeHomography"))
        .def("SetCoMValues",&QPHomographyRealLinear::SetCoMValues,args("x_position","y_position","z_position","x_speed", "y_speed"))
        .def("GetObjectiveFunctionValue",&QPHomographyRealLinear::GetObjectiveFunctionValue)
        ;

    class_<QPHomographyObstaclesLinear>("QPHomographyObstaclesLinear",init<std::string,FPTYPE,FPTYPE,FPTYPE,FPTYPE,FPTYPE,FPTYPE,FPTYPE,FPTYPE,FPTYPE,FPTYPE,FPTYPE,bool>() )
  
        .def("GetCoMAbsoluteX",&QPHomographyObstaclesLinear::GetCoMAbsoluteX)
        .def("GetCoMAbsoluteY",&QPHomographyObstaclesLinear::GetCoMAbsoluteY)
        .def("GetCoMAbsoluteZ",&QPHomographyObstaclesLinear::GetCoMAbsoluteZ)
        .def("GetCoMAbsoluteTheta",&QPHomographyObstaclesLinear::GetCoMAbsoluteTheta)
        .def("GetSupportFootAbsoluteX",&QPHomographyObstaclesLinear::GetSupportFootAbsoluteX)
        .def("GetSupportFootAbsoluteY",&QPHomographyObstaclesLinear::GetSupportFootAbsoluteY)
        .def("GetSupportFootAbsoluteZ",&QPHomographyObstaclesLinear::GetSupportFootAbsoluteZ)
        .def("GetNextSupportFootAbsoluteX",&QPHomographyObstaclesLinear::GetNextSupportFootAbsoluteX)
        .def("GetNextSupportFootAbsoluteY",&QPHomographyObstaclesLinear::GetNextSupportFootAbsoluteY)
        .def("GetSupportFootAbsoluteTheta",&QPHomographyObstaclesLinear::GetSupportFootAbsoluteTheta)
        .def("GetNextSupportFootAbsoluteTheta",&QPHomographyObstaclesLinear::GetNextSupportFootAbsoluteTheta)
        .def("SolveProblem",&QPHomographyObstaclesLinear::SolveProblem)
        .def("Continue",&QPHomographyObstaclesLinear::Continue)
        .def("Update",&QPHomographyObstaclesLinear::Update)
        .def("LogCurrentResults",&QPHomographyObstaclesLinear::LogCurrentResults)
        .def("SetCurrentHomography",&QPHomographyObstaclesLinear::SetCurrentHomography,args("h11","h12","h13","h21","h23","h31","h32","h33","isObstacle","isComputeHomography","c1","c2","c3","iteration"))
        .def("SetCoMValues",&QPHomographyObstaclesLinear::SetCoMValues,args("x_position","y_position","z_position","x_speed", "y_speed"))
        .def("GetObjectiveFunctionValue",&QPHomographyObstaclesLinear::GetObjectiveFunctionValue)
        .def("IsSupportFootChange",&QPHomographyObstaclesLinear::IsSupportFootChange)
    	;
    
	class_<QPHerdtSim>("QPHerdtSim",init<std::string>())
		.def("GetCoMAbsoluteX",&QPHerdtSim::GetCoMAbsoluteX)
		.def("GetCoMAbsoluteY",&QPHerdtSim::GetCoMAbsoluteY)
		.def("GetCoMAbsoluteZ",&QPHerdtSim::GetCoMAbsoluteZ)
		.def("GetCoMAbsoluteTheta",&QPHerdtSim::GetCoMAbsoluteTheta)
		.def("GetSupportFootAbsoluteX",&QPHerdtSim::GetSupportFootAbsoluteX)
		.def("GetSupportFootAbsoluteY",&QPHerdtSim::GetSupportFootAbsoluteY)
		.def("GetSupportFootAbsoluteZ",&QPHerdtSim::GetSupportFootAbsoluteZ)
		.def("GetNextSupportFootAbsoluteX",&QPHerdtSim::GetNextSupportFootAbsoluteX)
		.def("GetNextSupportFootAbsoluteY",&QPHerdtSim::GetNextSupportFootAbsoluteY)
		.def("GetSupportFootAbsoluteTheta",&QPHerdtSim::GetSupportFootAbsoluteTheta)
		.def("GetNextSupportFootAbsoluteTheta",&QPHerdtSim::GetNextSupportFootAbsoluteTheta)
		.def("SolveProblem",&QPHerdtSim::SolveProblem)
		.def("Continue",&QPHerdtSim::Continue)
		.def("Update",&QPHerdtSim::Update)
		.def("LogCurrentResults",&QPHerdtSim::LogCurrentResults)
		.def("SetCoMValues",&QPHerdtSim::SetCoMValues,args("x_position","y_position", "z_position","x_speed", "y_speed"))
		.def("GetObjectiveFunctionValue",&QPHerdtSim::GetObjectiveFunctionValue)
		;

	class_<QPHerdtReal>("QPHerdtReal",init<std::string,FPTYPE,FPTYPE,FPTYPE>())
		.def("GetCoMAbsoluteX",&QPHerdtReal::GetCoMAbsoluteX)
		.def("GetCoMAbsoluteY",&QPHerdtReal::GetCoMAbsoluteY)
		.def("GetCoMAbsoluteZ",&QPHerdtReal::GetCoMAbsoluteZ)
		.def("GetCoMAbsoluteTheta",&QPHerdtReal::GetCoMAbsoluteTheta)
		.def("GetSupportFootAbsoluteX",&QPHerdtReal::GetSupportFootAbsoluteX)
		.def("GetSupportFootAbsoluteY",&QPHerdtReal::GetSupportFootAbsoluteY)
		.def("GetSupportFootAbsoluteZ",&QPHerdtReal::GetSupportFootAbsoluteZ)
		.def("GetNextSupportFootAbsoluteX",&QPHerdtReal::GetNextSupportFootAbsoluteX)
		.def("GetNextSupportFootAbsoluteY",&QPHerdtReal::GetNextSupportFootAbsoluteY)
		.def("GetSupportFootAbsoluteTheta",&QPHerdtReal::GetSupportFootAbsoluteTheta)
		.def("GetNextSupportFootAbsoluteTheta",&QPHerdtReal::GetNextSupportFootAbsoluteTheta)
		.def("SolveProblem",&QPHerdtReal::SolveProblem)
		.def("Continue",&QPHerdtReal::Continue)
		.def("Update",&QPHerdtReal::Update)
		.def("LogCurrentResults",&QPHerdtReal::LogCurrentResults)
		.def("SetCoMValues",&QPHerdtReal::SetCoMValues,args("x_position","y_position", "z_position","x_speed", "y_speed"))
		.def("GetObjectiveFunctionValue",&QPHerdtReal::GetObjectiveFunctionValue)
		.def("SetCurrentReferenceSpeed",&QPHerdtReal::SetCurrentReferenceSpeed,args("x_speed_ref","y_speed_ref","theta_ref"))
		;
}