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

#include <string>
#include <memory>
#include <vector>
#include <ctime>
#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#ifdef USE_SPDLOG
#include <spdlog/spdlog.h>
#endif
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/exceptions.hpp>
#include <boost/algorithm/string.hpp>

#include "common.h"

namespace {

  const unsigned int TEN_MB = (1<<20)*10;
  const std::string LOG_ID = "basic_logger";

inline void logAnglesInfo(const int iteration, const FPTYPE f, const FPTYPE t, const std::string& tag)
{
    std::ostringstream oss;
    oss << "[" << tag << "]: "
        << "iteration=" << iteration
        << ", foot=" << f 
        << ", trunk=" << t;
    common::logString(oss.str());
}

inline void logGenericInfo(const int iteration, const FPTYPE x, const FPTYPE y, const std::string& tag)
{
    std::ostringstream oss;
    oss << "[" << tag << "]: "
        << "iteration=" << iteration
        << ", x=" << x 
        << ", y=" << y;
    common::logString(oss.str());
}

inline void logGenericInfo(const int iteration, const FPTYPE x, const FPTYPE y, const FPTYPE angle, const std::string& tag)
{
    std::ostringstream oss;
    oss << "[" << tag << "]: "
        << "iteration=" << iteration
        << ", x=" << x 
        << ", y=" << y 
        << ", orientation=" << angle;
        common::logString(oss.str());
}

inline void logGenericInfo(const int iteration, const int future_iteration, const FPTYPE x, const FPTYPE y, const FPTYPE angle, const std::string& tag)
{
    std::ostringstream oss;
    oss << "[" << tag << "]: "
        << "iteration=" << iteration
        << ", future_iteration=" << future_iteration        
        << ", x=" << x 
        << ", y=" << y 
        << ", orientation=" << angle;
        common::logString(oss.str());
}

inline void logGenericInfo(const int iteration, const int future_iteration, const FPTYPE x, const FPTYPE y, const std::string& tag)
{
    std::ostringstream oss;
    oss << "[" << tag << "]: "
        << "iteration=" << iteration
        << ", future_iteration=" << future_iteration        
        << ", x=" << x 
        << ", y=" << y; 
        common::logString(oss.str());
}

inline void logGenericInfo(const int iteration, const FPTYPE angle, const std::string& tag)
{
    std::ostringstream oss;
    oss << "[" << tag << "]: "
        << "iteration=" << iteration
        << ", orientation=" << angle;
        common::logString(oss.str());
}

} // namespace

void common::logString(const std::string &s) {
   char time_stamp[100];
   time_t now = std::time(0);
   std::strftime(time_stamp, sizeof(time_stamp), "%a %b %d %H:%M:%S %Y", std::localtime(&now));
   std::cout << "[" << std::string(time_stamp) << "] [info]" << s << std::endl;
}

boost::property_tree::ptree common::LoadParameters(const std::string config_file)
{
    boost::property_tree::ptree parameters;
    boost::property_tree::ini_parser::read_ini(config_file, parameters);
    return parameters;
}


std::vector<FPTYPE> common::ExpandRawRange(std::string raw_range)
{
    std::vector<std::string> all_ranges;
    std::vector<FPTYPE> expanded_range;
    boost::split(all_ranges, raw_range, boost::is_any_of(","));

    for (auto &current_range : all_ranges) {
        std::vector<std::string> value_times;
        FPTYPE value, times;
        boost::split(value_times, current_range, boost::is_any_of(":"));

        if (value_times.size() == 1) {
	  value = (FPTYPE)::atof(value_times[0].c_str());
            times = 1;
        } else if (value_times.size() == 2) {
            value = (FPTYPE)::atof(value_times[0].c_str());
            times = (FPTYPE)::atof(value_times[1].c_str());
        } else {
            std::cerr << "ERROR: could not expand range" << std::endl;
            throw std::exception();
        }

        for (int i=0; i<times; i++)
            expanded_range.push_back(value);
    }

    return expanded_range;
}

void common::logCoMPosition(const int iteration, const FPTYPE x, const FPTYPE y)
{
    logGenericInfo(iteration, x, y, "com_position");
}

void common::logPredictedCoMPosition(const int iteration, const int future_iteration, const FPTYPE x, const FPTYPE y, const FPTYPE t)
{
    logGenericInfo(iteration, future_iteration, x, y, t, "predicted_com_position");
}

void common::logPredictedZmpPosition(const int iteration, const int future_iteration, const FPTYPE x, const FPTYPE y)
{
    logGenericInfo(iteration, future_iteration, x, y, "predicted_zmp_position");
}

void common::logCoMSpeed(const int iteration, const FPTYPE x, const FPTYPE y)
{
    logGenericInfo(iteration, x, y, "com_speed");
}

void common::logCoMAcceleration(const int iteration, const FPTYPE x, const FPTYPE y)
{
    logGenericInfo(iteration, x, y, "com_acceleration");
}

void common::logCoMAngle(const int iteration, const FPTYPE angle)
{
    logGenericInfo(iteration, angle, "com_theta");
}

void common::logCoMAngularSpeed(const int iteration, const FPTYPE angular_speed)
{
    logGenericInfo(iteration, angular_speed, "com_angular_speed");
}

void common::logCoMAngularAcceleration(const int iteration, const FPTYPE angular_acceleration)
{
    logGenericInfo(iteration,angular_acceleration,"com_angular_acceleration");
}

void common::logFlyingFootAngle(const int iteration, const FPTYPE angle)
{
    logGenericInfo(iteration, angle, "flying_foot_theta");
}
void common::logSupportFootAngle(const int iteration, const FPTYPE angle)
{
    logGenericInfo(iteration, angle, "support_foot_theta");
}

void common::logJerks(const int iteration, const FPTYPE x, const FPTYPE y)
{
    logGenericInfo(iteration, x, y, "jerks");
}

void common::logAngleJerks(const int iteration, const FPTYPE f, const FPTYPE t)
{
    logAnglesInfo(iteration, f, t, "angle_jerks");
}

void common::logZMPPosition(const int iteration, const FPTYPE x, const FPTYPE y)
{
    logGenericInfo(iteration, x, y, "zmp_position");
}

void common::logFootPosition(const int iteration, const FPTYPE x, const FPTYPE y, const FPTYPE angle)
{
    logGenericInfo(iteration, x, y, angle, "foot_position");
}

void common::logPredictedFootPosition(const int iteration, const FPTYPE x, const FPTYPE y, const FPTYPE angle)
{
    logGenericInfo(iteration, x, y, angle, "predicted_foot_position");
}

void common::logDesiredPosition(const int iteration, const FPTYPE x, const FPTYPE y, const FPTYPE angle)
{
    logGenericInfo(iteration, x, y, angle, "desired_position");
}

void common::Log2DInformation(std::string tag, int iteration,  int ref_id, int pt_id, const Point2D_t &point, bool visible) {
    std::ostringstream oss;
    oss << tag << ": "
        << "iteration=" << iteration
        << ", reference_id=" << std::setw(3) << std::setfill('0') << ref_id
        << ", point_id=" << std::setw(3) << std::setfill('0') << pt_id
        << ", x=" << X(point)
        << ", y=" << Y(point)
        << ", visible=" << (visible?1:0);
    logString(oss.str());
}

std::vector<std::string> common::CommandLineParserLib::SetCommandLineOptions() {
    std::vector<std::string> keys;
    keys.push_back("reference.orientation0");
/*    keys.push_back("camera.fx");
    keys.push_back("camera.fy");
    keys.push_back("camera.u0");
    keys.push_back("camera.v0");
    keys.push_back("camera.sigma_noise");*/
    return keys;
}

void common::CommandLineParserLib::VerifiyCommandLineOptions(boost::property_tree::ptree &parameters) {

    std::vector<std::string> parameters_required_  = SetCommandLineOptions();

    for (const auto &parameter_name : parameters_required_) {
        if(has(parameter_name))
            parameters.put(parameter_name,get<FPTYPE>(parameter_name));
    }
}