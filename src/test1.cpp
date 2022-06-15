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
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>



#include "common.h"
#include "feet.h"
#include "formulations.h"
#include "models.h"
#include "qp.h"

namespace {

  const char MICROSECONDS[] = "microseconds";
  const char SECONDS[] = "seconds";

} // namespace

// Command-lines keys parser
const char* command_line_keys ={
        "{reference.orientation0  | <none> | desired orientation in radians}"
/*        "{camera.fx          | <none> | focal value in x  expressed in pixel units}"
        "{camera.fy          | <none> | focal value in y expressed in pixel units}"
        "{camera.u0          | <none> | principal point  }"
        "{camera.v0          | <none> | principal point  }"
        "{camera.sigma_noise | 2.0 | noise with a standard deviation of sigma_noise pixels}"
        "{qp.betah11          | 3.0 | gain of the element h11}"
        "{qp.betah12          | 1.0 | gain of the element h12}"
        "{qp.betah13          | 3.0 | gain of the element h13}"
        "{qp.betah31          | 2.0 | gain of the element h31}"
        "{qp.betah32          | 1.0 | gain of the element h32}"
        "{qp.betah33          | 0.8 | gain of the element h33}"*/
};


/******************************************************************************
 *
 *                                   main
 *
 *****************************************************************************/

int main(int argc, char* argv[])
{

    if (argc < 2) {
        std::cout << "ERROR: Invalid number of arguments provided" << std::endl
                  << "Usage:" << std::endl
                  << "\t./" + std::string(argv[0]) << " config_file.ini" << std::endl; 
        return -1;
    }

    SimulatorFactory sf;

    boost::filesystem::path config_file = std::string(argv[1]);

    std::cout << "parsing configuration file " << config_file << std::endl;
    auto parameters = common::LoadParameters(config_file.string());
    common::logString(": configuration read from: " + config_file.string());
//    std::cout << "parameters.get<FPTYPE>(\"camera.fx\"): " << parameters.get<FPTYPE>("camera.fx") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"camera.fy\"): " << parameters.get<FPTYPE>("camera.fy") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"camera.u0\"): " << parameters.get<FPTYPE>("camera.u0") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"camera.v0\"): " << parameters.get<FPTYPE>("camera.v0") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"camera.sigma_noise\"): " << parameters.get<FPTYPE>("camera.sigma_noise") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"reference.orientation0\"): " << parameters.get<FPTYPE>("reference.orientation0") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"qp.betah11\"): " << parameters.get<FPTYPE>("qp.betah11") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"qp.betah12\"): " << parameters.get<FPTYPE>("qp.betah12") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"qp.betah13\"): " << parameters.get<FPTYPE>("qp.betah13") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"qp.betah31\"): " << parameters.get<FPTYPE>("qp.betah31") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"qp.betah32\"): " << parameters.get<FPTYPE>("qp.betah32") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"qp.betah33\"): " << parameters.get<FPTYPE>("qp.betah33") << std::endl;
    if(argc>2){
        common::CommandLineParserLib parser(argc,argv,command_line_keys);
        parser.VerifiyCommandLineOptions(parameters);
    }
//    std::cout << "parameters.get<FPTYPE>(\"camera.fx\"): " << parameters.get<FPTYPE>("camera.fx") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"camera.fy\"): " << parameters.get<FPTYPE>("camera.fy") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"camera.u0\"): " << parameters.get<FPTYPE>("camera.u0") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"camera.v0\"): " << parameters.get<FPTYPE>("camera.v0") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"camera.sigma_noise\"): " << parameters.get<FPTYPE>("camera.sigma_noise") << std::endl;
//    std::cout << "parameters.get<FPTYPE>(\"reference.orientation0\"): " << parameters.get<FPTYPE>("reference.orientation0") << std::endl;


    try {
      auto simulator = sf.BuildSimulator(parameters);
      simulator->LogCurrentResults();      
      auto qp_start_time = boost::posix_time::microsec_clock::local_time();
      common::logString(" Creating solver...");
      QPSolver solver(dynamic_cast<QProblemInterface*>(simulator.get()));
      auto qp_end_time   = boost::posix_time::microsec_clock::local_time();
      auto duration = qp_end_time - qp_start_time;
      common::logString("[qp_time]: time=" + boost::lexical_cast<std::string>(duration.total_microseconds()) + " microseconds");

      // Main loop
      auto simulation_start_time = boost::posix_time::microsec_clock::local_time();

      while(simulator->Continue()) {
        simulator->LogCurrentResults(solver.GetSolution());

        // Uses the previous solution to update the simulator
        simulator->Update(solver.GetSolution());
	       
        // Solve the problem
        qp_start_time = boost::posix_time::microsec_clock::local_time();
        solver.SolveProblem();
        qp_end_time   = boost::posix_time::microsec_clock::local_time();
        auto duration = qp_end_time - qp_start_time;
        common::logString("[qp_time]: time=" + boost::lexical_cast<std::string>(duration.total_microseconds()) + " microseconds");
        common::logString("[objective_function]: " + boost::lexical_cast<std::string>(solver.GetObjectiveFunctionValue()));
        common::logString("[current_step_ratio]: " + boost::lexical_cast<std::string>(simulator->GetCurrentStepRatio()));
        common::logString("[remaining_steps]: " + boost::lexical_cast<std::string>(simulator->GetCurrentStepRemainingCycles()));
        common::logString("[support_type]: " + (simulator->GetCurrentSupportType()?std::string("single"):std::string("double")));
      }
      auto simulation_end_time = boost::posix_time::microsec_clock::local_time();
      auto simulation_duration = simulation_end_time - simulation_start_time;
      common::logString("[total_simulation_time]: time=" + boost::lexical_cast<std::string>(duration.total_seconds()) + " seconds");
      common::logString("[objective_function]: " + boost::lexical_cast<std::string>(solver.GetObjectiveFunctionValue()));
    } catch(std::exception &a) {
      std::cerr << a.what() << std::endl;
    } catch(...) {
      std::cerr << "Exception, had to quit..." << std::endl;
    }
   
    return 0;
}
