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

#ifndef COMMON_H
#define COMMON_H

#include <memory>
#include <string>
#include <vector>
#include <iomanip>

#include <Eigen/Dense>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <opencv2/opencv.hpp>

#ifdef USE_SPDLOG
#include <spdlog/spdlog.h>
#endif

#ifdef __USE_SINGLE_PRECISION__
typedef float FPTYPE;
#else
typedef double FPTYPE;
#endif

typedef Eigen::Vector3d Point3D_t;
typedef Eigen::Vector3d Vector3D_t;
typedef Eigen::Vector2d Point2D_t ;

typedef Eigen::Matrix<FPTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix_t;
typedef Eigen::Transform<FPTYPE, 3, Eigen::Affine> AffineTransformation;

// functions to get X, Y, Z coordinates
template <typename T>
inline FPTYPE& X(T& p) { return p(0); }

template <typename T>
inline FPTYPE& Y(T& p) { return p(1); }

inline FPTYPE& Z(Point3D_t& p) { return p(2); }

template <typename T>
inline const FPTYPE X(const T& p) { return p(0); }

template <typename T>
inline const FPTYPE Y(const T& p) { return p(1); }

inline const FPTYPE& Z(const Point3D_t& p) { return p(2); }



namespace common {

	const FPTYPE g = 9.81;

	boost::property_tree::ptree LoadParameters(const std::string config_file);

	// Logging stuff
	void init_logger(std::string tag);
#ifdef USE_SPDLOG
	std::shared_ptr<spdlog::logger> get_logger();
#endif
 	void close_logger();

	std::vector<FPTYPE> ExpandRawRange(std::string raw_range);
 	void logString(const std::string &s);
 	void logCoMPosition(const int iteration, const FPTYPE x, const FPTYPE y);
 	void logPredictedCoMPosition(const int iteration, const int future_iteration, const FPTYPE x, const FPTYPE y, const FPTYPE t);
 	void logPredictedZmpPosition(const int iteration, const int future_iteration, const FPTYPE x, const FPTYPE y);
 	void logCoMSpeed(const int iteration, const FPTYPE x, const FPTYPE y);
 	void logCoMAcceleration(const int iteration, const FPTYPE x, const FPTYPE y);
 	void logCoMAngle(const int iteration, const FPTYPE angle);
 	void logSupportFootAngle(const int iteration, const FPTYPE angle); 	
 	void logFlyingFootAngle(const int iteration, const FPTYPE angle);
 	void logCoMAngularSpeed(const int iteration, const FPTYPE angular_speed);
 	void logCoMAngularAcceleration(const int iteration, const FPTYPE angular_acceleration);
 	void logJerks(const int iteration, const FPTYPE x, const FPTYPE y);
 	void logAngleJerks(const int iteration, const FPTYPE f, const FPTYPE t); 	
 	void logZMPPosition(const int iteration, const FPTYPE x, const FPTYPE y);
 	void logFootPosition(const int iteration, const FPTYPE x, const FPTYPE y, const FPTYPE angle);
 	void logPredictedFootPosition(const int iteration, const FPTYPE x, const FPTYPE y, const FPTYPE angle);
 	void logDesiredPosition(const int iteration, const FPTYPE x, const FPTYPE y, const FPTYPE angle);
 	template<typename T>
	void Log3DInformation(std::string tag, int iteration, int ref_id, int pt_id, T point) {
	    std::ostringstream oss;
    	oss << tag << ": "
        	<< "iteration=" << iteration
        	<< ", reference_id=" << std::setw(3) << std::setfill('0') << ref_id
        	<< ", point_id=" << std::setw(3) << std::setfill('0') << pt_id        	
        	<< ", x=" << X(point)
        	<< ", y=" << Y(point)
        	<< ", z=" << Z(point);
    		common::logString(oss.str());
	}
	void Log2DInformation(std::string tag, int iteration,  int ref_id, int pt_id, const Point2D_t &point, bool visible=true);

	// Geometrical stuff
	inline FPTYPE YRotationAngleFromRotationMatrix(const Matrix_t& rotation_matrix) {
    	return std::atan(rotation_matrix(0, 2) / rotation_matrix(0, 0));
 	}

    class CommandLineParserLib : public cv::CommandLineParser
    {
        public:
            CommandLineParserLib(int argc, const char* const* argv, const char *key_map) : cv::CommandLineParser(argc,argv, key_map) {}
            ~CommandLineParserLib(){};
            void VerifiyCommandLineOptions(boost::property_tree::ptree &parameters);

        private:
        	std::vector<std::string> SetCommandLineOptions();

    };

} // namespace common

#endif  // COMMON_H
