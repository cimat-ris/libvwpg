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

#include <boost/lexical_cast.hpp>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "formulations.h"
#include "vision_utils.h"

void SimulatorFactory::VerifyParameters(const std::vector<std::string> &all_required_parameters, 
                                        const boost::property_tree::ptree &parameters)
{
    for (const auto &parameter_name : all_required_parameters)
        try {
            parameters.get<std::string>(parameter_name);
        } catch (boost::property_tree::ptree_bad_path e) {
            std::cerr << "ERROR: cannot start simulation missing parameter: \""
                      << parameter_name << "\" in configuration file." << std::endl;
            throw;
        }
}


void SimulatorFactory::LogInputParameters(const std::vector<std::string> &all_required_parameters, const boost::property_tree::ptree &parameters)
{
    for (const auto &parameter_name : all_required_parameters) {
      std::ostringstream oss;
      oss << "[parameter]: "
        << parameter_name << "="
        << parameters.get<std::string>(parameter_name);
      common::logString(oss.str());
    }
}

SimulatorFactory::SimulatorFactory() {
    std::vector<std::string> vherdt;
    vherdt.push_back("simulation.N");
    vherdt.push_back("simulation.m");
    vherdt.push_back("simulation.T");
    vherdt.push_back("simulation.robot");
    vherdt.push_back("simulation.double_support_lenght");
    vherdt.push_back("simulation.single_support_lenght");
    vherdt.push_back("qp.alpha");
    vherdt.push_back("qp.beta");
    vherdt.push_back("qp.gamma");
    vherdt.push_back("reference.x_com_speed");
    vherdt.push_back("reference.y_com_speed");
    vherdt.push_back("reference.orientation");
    vherdt.push_back("initial_values.foot_x_position");
    vherdt.push_back("initial_values.foot_y_position");
      
    std::vector<std::string> vhomog_sim;
    vhomog_sim.push_back("simulation.N");
    vhomog_sim.push_back("simulation.m");
    vhomog_sim.push_back("simulation.T");
    vhomog_sim.push_back("simulation.robot");
    vhomog_sim.push_back("simulation.double_support_lenght");
    vhomog_sim.push_back("simulation.single_support_lenght");
    vhomog_sim.push_back("simulation.nSimulatedPoints");
    vhomog_sim.push_back("qp.alpha");
    vhomog_sim.push_back("qp.betah11");
    vhomog_sim.push_back("qp.betah12");
    vhomog_sim.push_back("qp.betah13");
    vhomog_sim.push_back("qp.betah31");
    vhomog_sim.push_back("qp.betah32");
    vhomog_sim.push_back("qp.betah33");
    vhomog_sim.push_back("qp.gamma");
    vhomog_sim.push_back("qp.eta_x");
    vhomog_sim.push_back("qp.eta_y");
    vhomog_sim.push_back("qp.kappa");
    vhomog_sim.push_back("initial_values.foot_x_position");
    vhomog_sim.push_back("initial_values.foot_y_position");
    vhomog_sim.push_back("reference.camera_position0_x");
    vhomog_sim.push_back("reference.camera_position0_y");
    vhomog_sim.push_back("reference.orientation0");

    std::vector<std::string> vessential_sim;
    vessential_sim.push_back("simulation.N");
    vessential_sim.push_back("simulation.m");
    vessential_sim.push_back("simulation.T");
    vessential_sim.push_back("simulation.robot");
    vessential_sim.push_back("simulation.double_support_lenght");
    vessential_sim.push_back("simulation.single_support_lenght");
    vessential_sim.push_back("simulation.nSimulatedPoints");
    vessential_sim.push_back("qp.alpha");
    vessential_sim.push_back("qp.betae12");
    vessential_sim.push_back("qp.betae21");
    vessential_sim.push_back("qp.betae32");
    vessential_sim.push_back("qp.betae23");
    vessential_sim.push_back("qp.gamma");
    vessential_sim.push_back("qp.eta_x");
    vessential_sim.push_back("qp.eta_y");
    vessential_sim.push_back("qp.kappa");
    vessential_sim.push_back("initial_values.foot_x_position");
    vessential_sim.push_back("initial_values.foot_y_position");
    vessential_sim.push_back("reference.camera_position0_x");
    vessential_sim.push_back("reference.camera_position0_y");
    vessential_sim.push_back("reference.orientation0");
    
    requiredParameters_["herdt"]                          = vherdt;
    requiredParameters_["homography_simulated"]           = vhomog_sim;
    requiredParameters_["essential_simulated"]            = vessential_sim;
}

boost::shared_ptr<SimulatorInterface> SimulatorFactory::BuildSimulator(const boost::property_tree::ptree &parameters) {
    std::string formulation_name;

    try {
        formulation_name = parameters.get<std::string>("simulation.formulation"); 
    } catch (boost::property_tree::ptree_bad_path e) {
        std::cerr << "ERROR: parameter \"simulation.formulation\" must be specified" << std::endl;
        throw;
    }

    if (requiredParameters_.find(formulation_name) == requiredParameters_.end()) {
        std::cerr << "ERROR: unknown formulation \"" << formulation_name << "\"" << std::endl;
        throw std::exception();
    }
    LogInputParameters(requiredParameters_.at(formulation_name), parameters);
    VerifyParameters(requiredParameters_.at(formulation_name), parameters);

    if (formulation_name == "herdt") {
      boost::shared_ptr<SimulatorInterface> ptr(new Herdt(parameters));
      return ptr;
    } else if (formulation_name == "homography_simulated") {
        // Check if we will use a linear/non-linear solver
        try{
            if(parameters.get<std::string>("qp.linear")!="true"){
                boost::shared_ptr<SimulatorInterface> ptr(new HomographyNonLinearSimulated(parameters));
                return ptr;
            }
        } catch (boost::property_tree::ptree_bad_path e) {
            // Case the linear flag has not been found. Consider this as the linear case.
        }
      boost::shared_ptr<SimulatorInterface> ptr(new HomographySimulated(parameters));
      return ptr;
    } else if (formulation_name == "essential_simulated") {
      // Check if we will use a linear/non-linear solver
      try {
        if (parameters.get<std::string>("qp.linear")!="true") {
          boost::shared_ptr<SimulatorInterface> ptr(new EssentialNonLinearSimulated(parameters));
          return ptr;      
        }
      } catch (boost::property_tree::ptree_bad_path e) {
        // Case the linear flag has not been found. Consider this as the linear case.
      }
      boost::shared_ptr<SimulatorInterface> ptr(new EssentialSimulated(parameters));
      return ptr;
    } else {
      std::cerr << "ERROR: internal error constructor for  \"" << formulation_name << "\" is not available" << std::endl;
      throw std::exception();
    }
}



