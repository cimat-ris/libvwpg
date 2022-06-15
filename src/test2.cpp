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

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <string>

#include <boost/filesystem/path.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/chrono/include.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "common.h"
#include "feet.h"
#include "formulations.h"
#include "models.h"
#include "qp.h"

namespace {
  const char MICROSECONDS[] = "microseconds";
  const char SECONDS[] = "seconds";
} // namespace


/******************************************************************************
 *
 *                                   main
 *
 *****************************************************************************/

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cout << "ERROR: Invalid number of arguments provided" << std::endl
                  << "Usage:" << std::endl
                  << "\t./" + std::string(argv[0]) << " config_file.ini" << std::endl; 
        return -1;
    }
    
    boost::filesystem::path config_file = std::string(argv[1]);

    auto parameters = common::LoadParameters(config_file.string());
    common::logString(": configuration read from: " + config_file.string());
    try {

      // First homography
      Matrix_t h = Matrix_t::Identity(3,3);
      h(0,2)=0.1;
      auto simulator = boost::shared_ptr<SimulatorInterface>(new Homography(parameters,h(0,0),h(0,1),h(0,2),h(1,0),h(1,1),h(1,2),h(2,0),h(2,1),h(2,2)));
      auto qp_start_time = boost::posix_time::microsec_clock::local_time();
      QPSolver solver(dynamic_cast<QProblemInterface*>(simulator.get()));
      auto qp_end_time   = boost::posix_time::microsec_clock::local_time();
      auto duration = qp_end_time - qp_start_time;
      common::logString("[qp_time]: time=" + boost::lexical_cast<std::string>(duration.total_microseconds()) + " microseconds");
      
      // main loop
      std::cerr << "running homography simulation..." << std::endl;
      auto simulation_start_time = boost::posix_time::microsec_clock::local_time();

      while(simulator->Continue()) {
        // Here the solution would be applied to the robot
        simulator->Update(solver.GetSolution());
        simulator->LogCurrentResults(solver.GetSolution());
	
        // Launches the solver
        qp_start_time = boost::posix_time::microsec_clock::local_time();
        solver.SolveProblem();
        qp_end_time   = boost::posix_time::microsec_clock::local_time();
        auto duration = qp_end_time - qp_start_time;
        common::logString("[qp_time]: time=" + boost::lexical_cast<std::string>(duration.total_microseconds()) + " microseconds");
      }
      auto simulation_end_time = boost::posix_time::microsec_clock::local_time();
      auto simulation_duration = simulation_end_time - simulation_start_time;
      common::logString("[total_simulation_time]: time=" + boost::lexical_cast<std::string>(duration.total_seconds()) + " seconds");
      common::logString("[objective_function]: " + boost::lexical_cast<std::string>(solver.GetObjectiveFunctionValue()));    
    } catch(...) {
      std::cerr << "Exception, had to quit..." << std::endl;
    }
   
    return 0;
}
