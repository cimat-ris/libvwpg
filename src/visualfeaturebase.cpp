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
#include <time.h>
#include <random>

#include <boost/lexical_cast.hpp>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "formulations.h"
#include "vision_utils.h"

// Constructor
VisualFeatureBase::VisualFeatureBase(const boost::property_tree::ptree &parameters):
    FormulationBase(parameters),    
    alpha_(parameters.get<FPTYPE>("qp.alpha")),
    gamma_(parameters.get<FPTYPE>("qp.gamma")),
    eta_x_(parameters.get<FPTYPE>("qp.eta_x")),
    eta_y_(parameters.get<FPTYPE>("qp.eta_y")),
    kappa_(parameters.get<FPTYPE>("qp.kappa")),
    iterations_(parameters.get<int>("simulation.iterations")),
    min_points_to_estimate_constraint_(parameters.get<int>("simulation.nMinPointsToEstimateConstraint")),
    n_simulated_points_(parameters.get<int>("simulation.nSimulatedPoints")),
    first_reference_image_(0),
    last_reference_image_(0),
    number_of_samples_(0),
    push_com_(false),
    instant_switch_(false),
    switch_threshold_(0.15),
    d_(parameters.get<FPTYPE>("hidden.d")),
    c_x_com_(parameters.get<FPTYPE>("camera.c_x_com")),
    c_z_com_(parameters.get<FPTYPE>("camera.c_z_com")),
    camera_(parameters.get<FPTYPE>("camera.fx"),parameters.get<FPTYPE>("camera.fy"),
            parameters.get<FPTYPE>("camera.u0"),parameters.get<FPTYPE>("camera.v0"),
            parameters.get<FPTYPE>("camera.sigma_noise")),
    camera_ground_truth_(parameters.get<FPTYPE>("camera_ground_truth.fx"),parameters.get<FPTYPE>("camera_ground_truth.fy"),
                         parameters.get<FPTYPE>("camera_ground_truth.u0"),parameters.get<FPTYPE>("camera_ground_truth.v0"),
                         parameters.get<FPTYPE>("camera_ground_truth.sigma_noise")),
    world_flying_foot_orientation_(0.0),
    world_support_foot_orientation_(0.0),
    world_com_orientation_(0.0),
    world_next_support_foot_orientation_(0.0),
    thetaRef_(N_,1),
    u_0_(4 * N_ + 2 * m_,1),
    theta_ref_foot_(N_,1),
    u_0_prev(false),
    occlusion_start_(-1),
    occlusion_end_(-1),
    occlusion_policy_(InterestPoints::OcclusionPolicy::None),
    multiple_objective_method_(MultipleObjectives::WeightedAverages),
    occlusion_proportion_(0.0),
    x_speed_ref_(0.0)
{
    // Rotation controller.
    rotation_controller_ = boost::shared_ptr<RobotTrunkOrientationOptimizer>(new RobotTrunkOrientationOptimizer(N_, parameters.get<FPTYPE>("simulation.T"),m_,parameters.get<FPTYPE>("qp.alphaR"),parameters.get<FPTYPE>("qp.betaR"),parameters.get<FPTYPE>("qp.gammaR")));

    // Camera-to-com (will stay fixed)
    camera_T_com_ = Eigen::Translation<FPTYPE, 3>(-c_z_com_, c_x_com_, (  robot_physical_parameters_->camera_position() - robot_physical_parameters_->com_height() )) *
                    Eigen::AngleAxis<FPTYPE>(-M_PI_2, Point3D_t::UnitZ()) *
                    Eigen::AngleAxis<FPTYPE>(-M_PI_2, Point3D_t::UnitX());
    camera_T_com_ = camera_T_com_.inverse();

    // Initial world-to-com (used for displaying results)
    world_T_com_  = Eigen::Translation<FPTYPE, 3>(0, 0, robot_physical_parameters_->com_height());
    // Initial world-to-foot (used for displaying results)
    world_T_foot_ = Eigen::Translation<FPTYPE , 3>(0, 0, 0) * Eigen::AngleAxis<FPTYPE>(0.0, Point3D_t::UnitZ());
    // Initial
    com_T_foot_ = Eigen::Translation<FPTYPE , 3>(0, 0, -robot_physical_parameters_->com_height()) * Eigen::AngleAxis<FPTYPE>(0.0, Point3D_t::UnitZ());

    common::logString(": initialize current camera position data (used only in simulation)");
    // Initialize current position data at 0,0,z in world coordinates 
    Matrix_t camera_position_(3, 1);
    // TODO: generalize this to allow different initialization points
    camera_position_ << 0.0, 0.0, robot_physical_parameters_->camera_position();
    camera_.UpdateCameraPosition(camera_position_);
    camera_ground_truth_.UpdateCameraPosition(camera_position_);
    
    // Check if we have enough points to start with
    if (n_simulated_points_<min_points_to_estimate_constraint_)
        throw std::domain_error("ERROR: Not enough points to perform visual constraint estimation");
    // Optional parameters
    InitOptionalParameters(parameters);

    // Set u0 value to zero (used only in the non-linear case)
    u_0_.setZero();

    try {
        if(parameters.get<std::string>("simulation.u_O_prev")=="true"){
            u_0_prev = true;
        }
    } catch (boost::property_tree::ptree_bad_path e) {
        // Parameter has not been found. By default we consider this as false.
    }

}

// Constructor to use with python
VisualFeatureBase::VisualFeatureBase(const std::string nameFileParameters):
    FormulationBase(nameFileParameters),    
    alpha_(parameters_.get<FPTYPE>("qp.alpha")),
    gamma_(parameters_.get<FPTYPE>("qp.gamma")),
    eta_x_(parameters_.get<FPTYPE>("qp.eta_x")),
    eta_y_(parameters_.get<FPTYPE>("qp.eta_y")),
    kappa_(parameters_.get<FPTYPE>("qp.kappa")),
    iterations_(parameters_.get<int>("simulation.iterations")),
    min_points_to_estimate_constraint_(parameters_.get<int>("simulation.nMinPointsToEstimateConstraint")),
    n_simulated_points_(parameters_.get<int>("simulation.nSimulatedPoints")),
    first_reference_image_(0),
    last_reference_image_(0),
    number_of_samples_(0),
    push_com_(false),
    instant_switch_(false),
    switch_threshold_(0.15),
    d_(parameters_.get<FPTYPE>("hidden.d")),
    c_x_com_(parameters_.get<FPTYPE>("camera.c_x_com")),
    c_z_com_(parameters_.get<FPTYPE>("camera.c_z_com")),
    camera_(parameters_.get<FPTYPE>("camera.fx"),parameters_.get<FPTYPE>("camera.fy"),
            parameters_.get<FPTYPE>("camera.u0"),parameters_.get<FPTYPE>("camera.v0"),
            parameters_.get<FPTYPE>("camera.sigma_noise")),
    camera_ground_truth_(parameters_.get<FPTYPE>("camera_ground_truth.fx"),parameters_.get<FPTYPE>("camera_ground_truth.fy"),
                         parameters_.get<FPTYPE>("camera_ground_truth.u0"),parameters_.get<FPTYPE>("camera_ground_truth.v0"),
                         parameters_.get<FPTYPE>("camera_ground_truth.sigma_noise")),
    world_flying_foot_orientation_(0.0),
    world_support_foot_orientation_(0.0),
    world_com_orientation_(0.0),
    world_next_support_foot_orientation_(0.0),
    thetaRef_(N_,1),
    u_0_(4 * N_ + 2 * m_,1),
    theta_ref_foot_(N_,1),
    u_0_prev(false),
    occlusion_start_(-1),
    occlusion_end_(-1),
    occlusion_policy_(InterestPoints::OcclusionPolicy::None),
    multiple_objective_method_(MultipleObjectives::WeightedAverages),
    occlusion_proportion_(0.0),
    x_speed_ref_(0.0)
{
    common::logString(": initialize rotation controller");
    // Rotation controller.
    rotation_controller_ = boost::shared_ptr<RobotTrunkOrientationOptimizer>(new RobotTrunkOrientationOptimizer(N_, parameters_.get<FPTYPE>("simulation.T"),m_,parameters_.get<FPTYPE>("qp.alphaR"),parameters_.get<FPTYPE>("qp.betaR"),parameters_.get<FPTYPE>("qp.gammaR")));
    common::logString(": rotation controller initialized");

    // Camera-to-com (will stay fixed)
    camera_T_com_ = Eigen::Translation<FPTYPE, 3>(-c_z_com_, c_x_com_, (  robot_physical_parameters_->camera_position() - robot_physical_parameters_->com_height() )) *
                    Eigen::AngleAxis<FPTYPE>(-M_PI_2, Point3D_t::UnitZ()) *
                    Eigen::AngleAxis<FPTYPE>(-M_PI_2, Point3D_t::UnitX());
    camera_T_com_ = camera_T_com_.inverse();

    // Initial world-to-com (used for displaying results)
    world_T_com_  = Eigen::Translation<FPTYPE, 3>(0, 0, robot_physical_parameters_->com_height());
    // Initial world-to-foot (used for displaying results)
    world_T_foot_ = Eigen::Translation<FPTYPE , 3>(0, 0, 0) * Eigen::AngleAxis<FPTYPE>(0.0, Point3D_t::UnitZ());
    world_T_next_foot_ = world_T_foot_;
    // Initial
    com_T_foot_ = Eigen::Translation<FPTYPE , 3>(0, 0, -robot_physical_parameters_->com_height()) * Eigen::AngleAxis<FPTYPE>(0.0, Point3D_t::UnitZ());
    com_T_next_foot_  = com_T_foot_;

    common::logString(": initialize current camera position data (used only in simulation)");
    // Initialize current position data at 0,0,z in world coordinates 
    Matrix_t camera_position_(3, 1);
    // TODO: generalize this to allow different initialization points
    camera_position_ << 0.0, 0.0, robot_physical_parameters_->camera_position();
    camera_.UpdateCameraPosition(camera_position_);
    camera_ground_truth_.UpdateCameraPosition(camera_position_);
    
    // Check if we have enough points to start with
    if (n_simulated_points_<min_points_to_estimate_constraint_)
        throw std::domain_error("ERROR: Not enough points to perform visual constraint estimation");
    // Optional parameters
    InitOptionalParameters(parameters_);

    // Set u0 value to zero (used only in the non-linear case)
    u_0_.setZero();

    try {
        if(parameters_.get<std::string>("simulation.u_O_prev")=="true"){
            u_0_prev = true;
        }
    } catch (boost::property_tree::ptree_bad_path e) {
        // Parameter has not been found. By default we consider this as false.
    }

}

void VisualFeatureBase::UpdateReferenceImages_()
{
    if (number_of_samples_ < 20)
        return;

    bool update_reference_images = true;

    switch(multiple_objective_method_){
        case MultipleObjectives::WeightedAverages :

            // Check errors in visual features for current reference image
            for (const auto &visual_data_queue : latest_visual_data_) {
                const std::string name = visual_data_queue.first;
                const FPTYPE average = visual_data_queue.second;
                // Error computed as absolute value of the difference between average and expected values
                FPTYPE error = std::abs(average - expected_values[name]);

                std::ostringstream oss;
                oss << "[" << name << "_visual_errors]: "
                    << "average=" << average
                    << ", error=" << error;
                common::logString(oss.str());
                if (error > switch_threshold_)
                    update_reference_images = false;
            }
            break;
        case MultipleObjectives::SharedPredictionWindows:
            if (index_switch_threshold_ == false)
                    update_reference_images = false;
            break;
    }


    if (update_reference_images) {
        common::logString("[update_reference_image]: iteration=" + boost::lexical_cast<std::string>(current_iteration_));
        number_of_samples_ = 0;

        bool can_move_first_index  = (first_reference_image_ + 1) < last_reference_image_;
        bool can_move_second_index = (last_reference_image_ + 1) <= all_interest_points_.size();

        if (can_move_first_index && can_move_second_index) {
            ++first_reference_image_;
            ++last_reference_image_;
            TakePictureReferenceFromCurrentPosition();
            for (auto &visual_data_queue : latest_visual_data_)
                visual_data_queue.second = 0;
        } else if (can_move_first_index) {
            ++first_reference_image_;
            TakePictureReferenceFromCurrentPosition();
            for (auto &visual_data_queue : latest_visual_data_)
                visual_data_queue.second = 0;
        } else {
            common::logString(": Last Reference Image Reached");
            eta_x_ = 0.0;
        }
    }
}

// Initialization of optional parameters
void VisualFeatureBase::InitOptionalParameters(const boost::property_tree::ptree& parameters)
{
    // Push robot along y axis
    try {
        push_com_           = parameters.get<bool>("optional.y_push");
        push_com_intensity_ = parameters.get<FPTYPE>("optional.y_intensity");
        push_com_iteration_ = parameters.get<int>("optional.y_iteration");

    } catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ptree_bad_path> >) {
        std::ostringstream oss;
        oss << ": no optional push_com parameters received";
        common::logString(oss.str());
        push_com_ = false;
    }

    if (push_com_) {
        std::ostringstream oss;
        oss << ": perturbation will be applied at iteration " << push_com_iteration_;
        common::logString(oss.str());
    }

    // Occlusion intervals and policies
    try {
        occlusion_start_    = parameters.get<int>("camera.occlusion_start");
        occlusion_end_      = parameters.get<int>("camera.occlusion_end");
    } catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ptree_bad_path> >) {
        std::ostringstream oss;
        oss << ": no optional occlusion parameters received";
        common::logString(oss.str());
    }

    if (occlusion_start_>0 && occlusion_end_>occlusion_start_) {
        std::ostringstream oss;
        oss << ": occlusion will be applied from iteration " << occlusion_start_ << " to iteration " << occlusion_end_;
        common::logString(oss.str());
        try {
            occlusion_policy_   = static_cast<InterestPoints::OcclusionPolicy>(parameters.get<int>("camera.occlusion_policy"));
        } catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ptree_bad_path> >) {
            occlusion_policy_   = InterestPoints::OcclusionPolicy::None;
        }
        if (occlusion_policy_==InterestPoints::OcclusionPolicy::RandomlyOccluded) {
            try {
                occlusion_proportion_   = parameters.get<FPTYPE>("camera.occlusion_proportion");
            } catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ptree_bad_path> >) {
                occlusion_proportion_   = 0.0;
            }
        }
    }


    // Control transition between images
    try {
        instant_switch_ = parameters.get<bool>("optional.instant_switch");
    } catch  (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ptree_bad_path> >) {
        instant_switch_ = false;
    }

    if (instant_switch_)
        common::logString(" : Immediate transition between reference images");
    
    try {
        switch_threshold_ = parameters.get<FPTYPE>("optional.switch_threshold");
        common::logString(" : Updated switch_threshold_ from default 0.15 to " + boost::lexical_cast<std::string>(switch_threshold_));
    } catch  (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ptree_bad_path> >) {
        // nothing to do
    }

    try {
        multiple_objective_method_   = static_cast<MultipleObjectives>(parameters.get<int>("optional.multiple_objective_method"));
    } catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ptree_bad_path> >) {
        multiple_objective_method_   = MultipleObjectives::WeightedAverages;
    }

    try {
        x_speed_ref_   = parameters.get<FPTYPE>("optional.X_speed_ref");
    } catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ptree_bad_path> >) {
        x_speed_ref_   = 0.0;
    }

}

// Determine the weights associated to each reference image in the case of a sequence of images
std::vector<FPTYPE> VisualFeatureBase::GetReferenceWeights()
{
    int remaining_references = last_reference_image_ - first_reference_image_;
    std::vector<FPTYPE> v;
    
    if (instant_switch_) {
      v.push_back(1.0);
      v.push_back(0.0);
      v.push_back(0.0);
      return v;
    }
    
    if (remaining_references >= 3) {
      v.push_back(0.85);
      v.push_back(0.10);
      v.push_back(0.05);
      return v;
    }
    
    if (remaining_references == 2) {
      common::logString(": two reference images remaining ");
      v.push_back(0.85);
      v.push_back(0.15);
      return v;
    }
    
    v.push_back(1.0);
    return v;
}

// This reads the parameters data in search of 3d reference points to be used
std::vector<Point3D_t> VisualFeatureBase::Load3DReferencePoints_(
                            const boost::property_tree::ptree &parameters,
                            const AffineTransformation &com_T_world,
                            const int targetId)
{
    std::vector<Point3D_t> points;
    std::string suffix_id = "reference.target" + boost::lexical_cast<std::string>(targetId);
    std::string name_target = parameters.get<std::string>(suffix_id);
    std::string log_entry(": found a reference image named "+ name_target);
    common::logString(log_entry);

    common::logDesiredPosition(current_iteration_,parameters.get<FPTYPE>("reference.camera_position" +  boost::lexical_cast<std::string>(targetId) +"_x"),
                               parameters.get<FPTYPE>("reference.camera_position" + boost::lexical_cast<std::string>(targetId) + "_y"),
                               parameters.get<FPTYPE>("reference.orientation"+  boost::lexical_cast<std::string>(targetId)) );

    std::random_device rd;
    std::mt19937 gen(rd());
    std::string formulation = parameters.get<std::string>("simulation.formulation");
    std::string ymin = "reference.p_y_min"+ boost::lexical_cast<std::string>(targetId);
    std::string ymax = "reference.p_y_max"+ boost::lexical_cast<std::string>(targetId);
    std::uniform_real_distribution<> dis_y(parameters.get<FPTYPE>(ymin), parameters.get<FPTYPE>(ymax));
    std::string zmin = "reference.p_z_min"+ boost::lexical_cast<std::string>(targetId);
    std::string zmax = "reference.p_z_max"+ boost::lexical_cast<std::string>(targetId);
    std::uniform_real_distribution<> dis_z(parameters.get<FPTYPE>(zmin), parameters.get<FPTYPE>(zmax));

    if( formulation == "homography_simulated"){
        // The plane is at dis_x
        std::string xmin = "reference.p_x_min"+ boost::lexical_cast<std::string>(targetId);
        const int dis_x = parameters.get<FPTYPE>(xmin);
        for (int suffix = 0; suffix<n_simulated_points_; suffix++) {
            Point3D_t p0(dis_x, dis_y(gen), dis_z(gen));
            common::Log3DInformation("[reference_world_point_at_world_frame]", current_iteration_, targetId, suffix, p0);
            // Points are read in the world frame and transformed into the com frame
            p0 = com_T_world * p0;
            points.push_back(p0);
        }
    }
    else{ //Essential case
        std::string xmin = "reference.p_x_min"+ boost::lexical_cast<std::string>(targetId);
        std::string xmax = "reference.p_x_max"+ boost::lexical_cast<std::string>(targetId);
        std::uniform_real_distribution<> dis_x(parameters.get<FPTYPE>(xmin), parameters.get<FPTYPE>(xmax));
        // Reads nSimulatedPoints_ points
        for (int suffix = 0; suffix<n_simulated_points_; suffix++) {
            Point3D_t p0(dis_x(gen), dis_y(gen), dis_z(gen));
            common::Log3DInformation("[reference_world_point_at_world_frame]", current_iteration_, targetId, suffix, p0);
            // Points are read in the world frame and transformed into the com frame
            p0 = com_T_world * p0;
            points.push_back(p0);
        }
    }

    return points;
}

void VisualFeatureBase::InitReferenceImage_(
                     const boost::property_tree::ptree &parameters,
                     const std::vector<Point3D_t> &reference_points_in_world_coordinates,
                     const int reference_id,
                     bool virtualImage,
                     Camera camera_copy,
                     bool groundTruth)
{

     // Camera target position/orientation is read in the .init file
    std::string camera_id(boost::lexical_cast<std::string>(reference_id));
    FPTYPE camera_x_position  = parameters.get<FPTYPE>("reference.camera_position" + camera_id + "_x");
    FPTYPE camera_y_position  = parameters.get<FPTYPE>("reference.camera_position" + camera_id + "_y");
    FPTYPE camera_orientation = parameters.get<FPTYPE>("reference.orientation"     + camera_id);

    // Camera init position is supposed to be zero (0,0,z).
    // TODO: change it to allow for more experiments to be possible
    Matrix_t camera_position(3, 1);
    camera_position << 0.0, 0.0, robot_physical_parameters_->camera_position();
    camera_copy.UpdateCameraPosition(camera_position);

    // Compute the CoM-Reference-to-CoM-Init transform based on the camera target position
    // This transform takes points in the CoM-Reference (target) frame and maps them to the CoM-Init frame.
    // TODO: change it to allow for more experiments to be possible, as it supposes the initial position is always zero.
    AffineTransformation com_T_com_at_reference;
    com_T_com_at_reference = Eigen::Translation<FPTYPE, 3>(camera_x_position, camera_y_position, 0.0);
    com_T_com_at_reference = com_T_com_at_reference * Eigen::AngleAxis<FPTYPE>(camera_orientation, Point3D_t::UnitZ());

    // This other transform takes inversely points in the CoM-Init frame (zero) and maps them to the CoM-Reference frame.
    auto com_at_reference_T_com = com_T_com_at_reference.inverse();

    // Map the 3D reference points from the CoM-Init frame to the CoM-Reference frame
    std::vector<Point3D_t> points_in_com_coordinates; 
    for (auto &point : reference_points_in_world_coordinates)
        points_in_com_coordinates.push_back(com_at_reference_T_com * point);

    if(!groundTruth) {
        // Interest points are formed here; in the structure, we store the current and reference 3d and image coordinates
        // points_in_com_coordinates             are the points in the CoM target position
        // reference_points_in_world_coordinates are the points in the CoM init (current) position
        all_interest_points_.push_back(InterestPoints(camera_copy, points_in_com_coordinates, reference_points_in_world_coordinates,true));
        const std::vector<Point3D_t> &reference_world_point = all_interest_points_.back().GetInReferenceWorldCoordinates();
        for (int k=0;k<reference_world_point.size();k++)
            common::Log3DInformation("[reference_world_point]", current_iteration_, reference_id, k, reference_world_point[k]);
        const std::vector<Point2D_t> &reference_image_points = all_interest_points_.back().GetInReferenceImageCoordinates();
        for (int k = 0; k < reference_image_points.size(); k++)
            common::Log2DInformation("[reference_image_point]", current_iteration_, reference_id, k, reference_image_points[k]);
    }
    else{
        // Interest points for the ground truth are formed here; in the structure is stored the current and reference 3d and image coordinates
        // points_in_com_coordinates             are the points in the CoM target position
        // reference_points_in_world_coordinates are the points in the CoM init (current) position
        all_ground_truth_interest_points_.push_back(InterestPoints(camera_copy, points_in_com_coordinates, reference_points_in_world_coordinates));
        const std::vector<Point3D_t> &reference_world_point_gt = all_ground_truth_interest_points_.back().GetInReferenceWorldCoordinates();
        for (int k=0;k<reference_world_point_gt.size();k++)
            common::Log3DInformation("[reference_world_point_gt]", current_iteration_, reference_id, k, reference_world_point_gt[k]);
        const std::vector<Point2D_t> &reference_image_points_gt = all_ground_truth_interest_points_.back().GetInReferenceImageCoordinates();
        for (int k=0;k<reference_image_points_gt.size();k++)
            common::Log2DInformation("[reference_image_point_gt]", current_iteration_, reference_id, k, reference_image_points_gt[k]);
    }

    // In the case we use a virtual image (e.g., with essential matrices, we do the same as above from a virtual viewing position)
    if (virtualImage) {
        if (!groundTruth){
            camera_position << 0.0, 0.0, robot_physical_parameters_->camera_position()+parameters.get<FPTYPE>("reference.virtual_height"+camera_id);
            camera_copy.UpdateCameraPosition(camera_position);

            // Interest points are formed here; in the structure is stored the current and reference 3d and image coordinates
            // points_in_com_coordinates             are the points in the CoM target position
            // reference_points_in_world_coordinates are the points in the CoM init (current) position
            all_virtual_interest_points_.push_back(InterestPoints(camera_copy, points_in_com_coordinates, reference_points_in_world_coordinates,true));

            const std::vector<Point2D_t> &virtual_reference_image_points = all_virtual_interest_points_.back().GetInReferenceImageCoordinates();
            for (int k=0;k<virtual_reference_image_points.size();k++)
                common::Log2DInformation("[virtual_reference_image_point]", current_iteration_, reference_id, k, virtual_reference_image_points[k]);
        }
        else{
            camera_position << 0.0, 0.0, robot_physical_parameters_->camera_position()+parameters.get<FPTYPE>("reference.virtual_height"+camera_id);
            camera_copy.UpdateCameraPosition(camera_position);

            // Interest points are formed here; in the structure is stored the current and reference 3d and image coordinates
            // points_in_com_coordinates             are the points in the CoM target position
            // reference_points_in_world_coordinates are the points in the CoM init (current) position
            all_ground_truth_virtual_interest_points_.push_back(InterestPoints(camera_copy, points_in_com_coordinates, reference_points_in_world_coordinates));

            const std::vector<Point2D_t> &virtual_reference_image_points_gt = all_ground_truth_virtual_interest_points_.back().GetInReferenceImageCoordinates();
            for (int k=0;k<virtual_reference_image_points_gt.size();k++)
                common::Log2DInformation("[virtual_reference_image_point_gt]", current_iteration_, reference_id, k, virtual_reference_image_points_gt[k]);

        }
    }
}

// Log the image positions of the reference points from the irst reference image in the current (initial) position
void VisualFeatureBase::TakePictureReferenceFromCurrentPosition()
{
    const std::vector<Point3D_t>& reference_interest_points = all_interest_points_[first_reference_image_].GetInCurrentWorldCoordinates();

    for (int k=0;k<reference_interest_points.size();k++) {
        Matrix_t image_points = camera_.TakePictureImageCoordinates(reference_interest_points[k]); 
        common::Log2DInformation("[reference_image_from_initial_position]", current_iteration_, first_reference_image_, k, image_points);
    }
}

// Log the predictions. 
void VisualFeatureBase::LogCurrentPredictions(const Matrix_t &solution) const
{
    // Every x iterations, log the whole predicted CoM trajectory
    if (current_iteration_%10==9) {
        // Rebuild the predicted CoM trajectory based on the solution
        const Matrix_t &Ppu = mpc_model_.get_Ppu();
        const Matrix_t &Pps = mpc_model_.get_Pps();
        const Matrix_t &Pzu = mpc_model_.get_Pzu();
        const Matrix_t &Pzs = mpc_model_.get_Pzs();
        const Matrix_t &xcom_state = dynamic_model_.GetXStateVector();
        const Matrix_t &ycom_state = dynamic_model_.GetYStateVector();    
        Matrix_t Xk1   = Pps*xcom_state + Ppu*(u_0_.block(  0      , 0, N_, 1)+solution.block(  0      , 0, N_, 1));
        Matrix_t Yk1   = Pps*ycom_state + Ppu*(u_0_.block(  N_+  m_, 0, N_, 1)+solution.block(  N_+  m_, 0, N_, 1));
        Matrix_t ZXk1  = Pzs*xcom_state + Pzu*(u_0_.block(  0      , 0, N_, 1)+solution.block(  0      , 0, N_, 1));
        Matrix_t ZYk1  = Pzs*ycom_state + Pzu*(u_0_.block(  N_+  m_, 0, N_, 1)+solution.block(  N_+  m_, 0, N_, 1));

        Matrix_t thetas;
        // Non linear case, the orientations are optimized in the same QP as the translations
        if (solution.rows()>2*N_+2*m_) {
            const Matrix_t &tcom_state = dynamic_model_.GetTCOMStateVector();  
            // These are orientations relative to the current CoM frame  
            thetas = Pps*tcom_state + Ppu*(u_0_.block(3*N_+2*m_, 0, N_, 1)+solution.block(3*N_+2*m_, 0, N_, 1));
        }
        // Linear case, the orientations are given by the rotation controller
        else {
            // These are orientations relative to the current CoM frame  
            thetas = rotation_controller_->GetTrunkOrientation();
        }
        for (int i=0;i<N_;i++) {
            // At that point, everything is given in the k-1 frame (dynamical system has not been updated)
            Point3D_t compoint( Xk1(i,0), Yk1(i,0), -robot_physical_parameters_->com_height());
            Point3D_t zmppoint(ZXk1(i,0),ZYk1(i,0), -robot_physical_parameters_->com_height());
            compoint = world_T_com_ * compoint;
            zmppoint = world_T_com_ * zmppoint;            
            common::logPredictedCoMPosition(current_iteration_, i, X(compoint), Y(compoint), world_com_orientation_+thetas(i,0));
            common::logPredictedZmpPosition(current_iteration_, i, X(zmppoint), Y(zmppoint));            
        }
    }
}

// Log the CoM position in the world frame, and the speed
void VisualFeatureBase::LogCurrentResults(const Matrix_t &solution) const
{
    if (current_iteration_==0)
        return;
    std::ostringstream oss;
    // Position of the CoM: zero
    Point3D_t point(0.0,0.0,0.0);
    // The positions are mapped to the world frame for proper display
    point = world_T_com_ * point;
    // Log CoM absolute position
    common::logCoMPosition(current_iteration_, X(point), Y(point));
    // Log CoM absolute orientation
    common::logCoMAngle(current_iteration_,  world_com_orientation_);
    // Log flying foot absolute orientation
    common::logFlyingFootAngle(current_iteration_, world_flying_foot_orientation_);    
    // Log support foot absolute orientation
    common::logSupportFootAngle(current_iteration_, world_support_foot_orientation_);   
    // Log current speed (in local frame) 
    common::logCoMSpeed(current_iteration_, 
                        dynamic_model_.GetCoM_X_Speed(),
                        dynamic_model_.GetCoM_Y_Speed());
    // Log current acceleration (in local frame) 
    common::logCoMAcceleration(current_iteration_, 
                                dynamic_model_.GetCoM_X_Acceleration(),
                                dynamic_model_.GetCoM_Y_Acceleration());

    // No solution for this step
    if (!solution.rows())
        return;

    // ZMP information
    Point3D_t zmp(dynamic_model_.GetZMP_X_Position(), 
                  dynamic_model_.GetZMP_Y_Position(), -robot_physical_parameters_->com_height());

    // ZMP in the new CoM frame
    // mk_T_mk1 is the transform taking coordinates in the new CoM and mapping to the previous one
    // it needs to be computed by the simulation step
    zmp = mk_T_mk1.inverse() * zmp;
    // ZMP in the world frame
    zmp = world_T_com_ * zmp;
    common::logZMPPosition(current_iteration_, X(zmp), Y(zmp));

    // Get the first step of the solution, the jerks in x, y, tcom and tfoot    
    const FPTYPE &x_jerk = solution(0    , 0);
    const FPTYPE &y_jerk = solution(N_+m_, 0);
    common::logJerks(current_iteration_, x_jerk, y_jerk);

    if (solution.rows()>2*N_+2*m_) {
        // When this makes sense, print the jerks in angles
        const FPTYPE &f_jerk = solution(2*N_+2*m_, 0);
        const FPTYPE &t_jerk = solution(3*N_+2*m_, 0);
        common::logAngleJerks(current_iteration_, f_jerk, t_jerk);
    } else{
        FPTYPE comAngularSpeed = rotation_controller_->GetCoMAngularSpeed();
        FPTYPE comAngularAcceleration = rotation_controller_->GetCoMAngularAcceleration();
        common::logCoMAngularSpeed(current_iteration_,comAngularSpeed);
        common::logCoMAngularAcceleration(current_iteration_,comAngularAcceleration);
    }
    // Log the predicted foot positions
    // TODO: is rotation_controller_->GetFeetOrientation() an absolute rotation or relative one?
    const Matrix_t &reference_orientation = rotation_controller_->GetFeetOrientation();
    for (int i=0; i<m_; i++) {
        Point3D_t point(solution(N_ + i, 0), solution(N_ + m_ + N_ + i, 0), -robot_physical_parameters_->com_height());
        point = world_T_com_ * point;
        common::logPredictedFootPosition(current_iteration_, X(point), Y(point), reference_orientation(m_,0));
    }
    // Log support foot position whenever there is a change on it
    if (!step_generator_.IsSameSupportFoot()) {
        Point3D_t point(support_foot_position_(0), support_foot_position_(1), -robot_physical_parameters_->com_height());
        point = world_T_com_ * point;
        common::logFootPosition(current_iteration_, X(point), Y(point), world_support_foot_orientation_ );
    }
}

// Main update function. Receives the solution computed in the previous iteration.
void VisualFeatureBase::Update(const Matrix_t &solution) {
    // Predictions. They will be used in the next iteration to compute the model error.
    UpdatePredictedValues(solution);

    // Update simulation based on the previous solution. ``Moves'' the dynamical system according to the computed solution.
    UpdateSimulation(solution);

    // Prepare data for the next optimization step
    PrepareDataForNextOptimization(solution);
}


// Main simulation update function. Applies the computed solution to the simulated system.
// Updates the dynamical model
void VisualFeatureBase::UpdateSimulation(const Matrix_t &solution)
{
    // In the linear case, u_0_ should be simply zero.
    Matrix_t solution_u0;
    // Non-linear case
    if (solution.rows()>2*N_+2*m_) 
        solution_u0 = solution + u_0_;
    else
        solution_u0 = solution;

    // Get the first step of the solution, the jerks in x, y, tcom and tfoot.
    // They are applied immediately to the dynamic model.    
    FPTYPE x_jerk = solution_u0(0    , 0);
    FPTYPE y_jerk = solution_u0(N_+m_, 0);
    FPTYPE footorientation = 0.0;
    FPTYPE dfootorientation= 0.0;
    FPTYPE comorientation  = 0.0;
    FPTYPE nextFootOrientation = 0.0;

    // Perturbation
    if (push_com_ && current_iteration_ == push_com_iteration_)
        y_jerk *= push_com_intensity_;


    // Occlusion
    if (current_iteration_==occlusion_start_){
        all_interest_points_[0].SetOcclusionPolicy(occlusion_policy_,occlusion_proportion_);
        all_virtual_interest_points_[0].SetOcclusionPolicy(occlusion_policy_,occlusion_proportion_);
    }
    if (current_iteration_==occlusion_end_){
        all_interest_points_[0].SetOcclusionPolicy(InterestPoints::OcclusionPolicy::None);
        all_virtual_interest_points_[0].SetOcclusionPolicy(InterestPoints::OcclusionPolicy::None);
    }


    // Update the internal parameters of the step generator
    // May toggle the support foot
    step_generator_.UpdateSequence();

    if (solution.rows()>2*N_+2*m_) {
        // Non-linear case.
        // Update the dynamic model with the computed jerks.
        // After this the dynamic model contains the updated positions in the previous CoM frame
        FPTYPE tfoot_jerk = solution_u0(2*N_+2*m_, 0);
        FPTYPE tcom_jerk  = solution_u0(3*N_+2*m_, 0);
        dynamic_model_ .UpdateState(x_jerk, y_jerk, tcom_jerk, tfoot_jerk);

        // Flying foot and CoM orientations (in the previous CoM frame)
        footorientation = dynamic_model_.GetTFOOT_Position();
        comorientation  = dynamic_model_.GetTCOM_Position();

    } else {
        // Linear case.
        // Update the dynamic model with the computed jerks.
        // After this the dynamic model contains the updated positions in the previous CoM frame
        dynamic_model_.UpdateState(x_jerk, y_jerk);
        // Flying foot and CoM orientations (in the previous CoM frame)
        footorientation     = rotation_controller_->GetFeetOrientation()(0,0);
        comorientation      = rotation_controller_->GetTrunkOrientation()(0,0);
        nextFootOrientation = rotation_controller_->GetFeetOrientation()(7,0);
    }

    
    // Update the flying foot orientation: it is equal to the orientation of the previous CoM frame + footorientation
    world_flying_foot_orientation_       = world_com_orientation_ + footorientation;

    if (!step_generator_.IsSameSupportFoot()) {
        common::logString(": change support foot ");
        // This is the new support foot position, expressed in the previous CoM frame
        X(support_foot_position_) = solution_u0(N_, 0);
        Y(support_foot_position_) = solution_u0(N_ + m_ + N_, 0);
        // Determines the transform from the new support foot to the previous CoM frame  
        // The orientation is the one the currently flying foot has (but that will be reset) 
        com_T_foot_ =   Eigen::Translation<FPTYPE , 3>(X(support_foot_position_), Y(support_foot_position_), -robot_physical_parameters_->com_height()) *
                        Eigen::AngleAxis<FPTYPE>(footorientation, Point3D_t::UnitZ());
        // world_T_com_ is the transformation from the previous CoM frame to the World
        // We deduce the transformation from the new support foot to the World
        world_T_foot_ = world_T_com_ * com_T_foot_;
        // Log flying foot absolute orientation
        common::logSupportFootAngle(current_iteration_+1, world_support_foot_orientation_);   
        common::logFlyingFootAngle(current_iteration_ +1, world_flying_foot_orientation_); 
        world_flying_foot_orientation_  = world_support_foot_orientation_;
        // Support foot orientation is updated here and in PrepareDataForNextOptimization
        dynamic_model_.ResetFlyingFootState();

        // Deduce the new support foot world orientation from world_T_foot_
        world_support_foot_orientation_ = -std::atan2(world_T_foot_(0,1),world_T_foot_(0,0));

        com_T_next_foot_ =   Eigen::Translation<FPTYPE , 3>(X(support_foot_position_), Y(support_foot_position_), -robot_physical_parameters_->com_height()) *
                        Eigen::AngleAxis<FPTYPE>(nextFootOrientation, Point3D_t::UnitZ());
        // world_T_com_ is the transformation from the previous CoM frame to the World
        // We deduce the transformation from the new support foot to the World
        world_T_next_foot_ = world_T_com_ * com_T_next_foot_;
        world_next_support_foot_orientation_ = -std::atan2(world_T_next_foot_(0,1),world_T_next_foot_(0,0));

        if (solution.rows()<=2*N_+2*m_)// linear case
            rotation_controller_->SetSupportFootOrientation(footorientation);

        // Proceeed to update the reference images
        UpdateReferenceImages_();
    }

    // Transform taking coordinates in the new CoM and mapping to the previous one
    mk_T_mk1 = Eigen::Translation<FPTYPE, 3>(dynamic_model_.GetCoM_X_Position(), 
                                             dynamic_model_.GetCoM_Y_Position(), 0.0) *
                                            Eigen::AngleAxis<FPTYPE>(comorientation, Eigen::Vector3d::UnitZ());
    // Update com to world         
    world_T_com_ = world_T_com_ * mk_T_mk1;
    // Update the world_com_orientation (for simulation display only)
    world_com_orientation_  += comorientation;

    // Here the 3D points are re-expressed relatively to the current position  
    AffineTransformation mk1_T_mk = mk_T_mk1.inverse();
    for (auto &points : all_virtual_interest_points_)
        points.UpdateCurrentWorldCoordinates(mk1_T_mk);
    for (auto &points : all_ground_truth_virtual_interest_points_)
        points.UpdateCurrentWorldCoordinates(mk1_T_mk);
    for (auto &points : all_interest_points_)
        points.UpdateCurrentWorldCoordinates(mk1_T_mk);
    for (auto &points : all_ground_truth_interest_points_)
        points.UpdateCurrentWorldCoordinates(mk1_T_mk);
    // Update visual constraint matrices
    SimulateVisualFeatures();

    ++current_iteration_;
}

// Build a vector a reference angles to be reached
void VisualFeatureBase::RefAnglesInterpolation(FPTYPE ref_angle)
{
    thetaRef_.setZero();
    theta_ref_foot_.setZero();

    FPTYPE orientation = ref_angle > 0 ? std::min(SOFT_ANGLE, ref_angle) : std::max(-SOFT_ANGLE, ref_angle);
    const FPTYPE delta = orientation / ((FPTYPE)N_ - 1);

    for (int i = 0; i < N_; i++)
        thetaRef_(i) = (i+1) * delta;

    //Reference for the flying foot.
    Matrix_t Uc = step_generator_.GetCurrent_U_Matrix();
    Matrix_t Uf = step_generator_.GetFuture_U_Matrix();
    unsigned short int s1 = step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
    unsigned short int s2 = step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));
    const FPTYPE delta2 = orientation / ((FPTYPE)(N_-(s2-s1)) - 1);

    for(int i=0; i<s1;i++)
        theta_ref_foot_(i) = (i+1)*delta2;

    for(int i=s1; i<s2;i++)
        theta_ref_foot_(i) = (i+1-s1)*delta2;

    for(int i=s2; i<N_;i++)
        theta_ref_foot_(i) = (i-(s2-s1))*delta2;
}

// Get the linear matrix for inequalities
Matrix_t VisualFeatureBase::nlGetA()
{
    Matrix_t A(3 * N_ + 2 * m_, 4 * N_ + 2 * m_);
    A.setZero();
    const Matrix_t &Ppu = mpc_model_.get_Ppu();
    const Matrix_t &Pzu = mpc_model_.get_Pzu();
    const Matrix_t &Pzs = mpc_model_.get_Pzs();
    const Matrix_t &Pps = mpc_model_.get_Pps();
    const Matrix_t &Ppuk= mpc_model_.get_Ppuk();
    const Matrix_t &tV  = mpc_model_.get_ThetaV();

    Matrix_t U_future  = step_generator_.GetFuture_U_Matrix();
    Matrix_t U_current = step_generator_.GetCurrent_U_Matrix();

    const Matrix_t &x_state = dynamic_model_.GetXStateVector();
    const Matrix_t &y_state = dynamic_model_.GetYStateVector();

    const Matrix_t &jerks_x = u_0_.block(0,0,N_,1);
    const Matrix_t &jerks_y = u_0_.block(N_ + m_,0,N_,1);
    const Matrix_t &jerks_tf= u_0_.block(2*N_+2*m_,0,N_,1);
    const Matrix_t &Xf      = u_0_.block(N_,0,m_,1);
    const Matrix_t &Yf      = u_0_.block(2 * N_ + m_,0,m_,1);

    Matrix_t thetasFoot = tV + Ppuk*jerks_tf;

    // For ZMP restrictions.
    Matrix_t Uc = step_generator_.GetCurrent_U_Matrix();
    Matrix_t Uf = step_generator_.GetFuture_U_Matrix();
    unsigned short int s1 = step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
    unsigned short int s2 = step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));
    Matrix_t SmVector(m_+1,1); SmVector.setZero();
    SmVector(0,0) = s1; SmVector(1,0) = s2; SmVector(2,0) = N_;
    Matrix_t thetas_support(m_+1,1); thetas_support.setZero();
    thetas_support(0,0) = dynamic_model_.GetTFOOTSupport();
    thetas_support(1,0) = thetasFoot(s1-1,0);
    thetas_support(2,0) = thetasFoot(s2-1,0);

    // For ZMP constraints (see Noe's C1)
    A.block(0,0,2 * N_, 4 * N_ + 2 * m_) = robot_restrictions_.GetAZMPNolinealRestrictions(thetas_support,Pzs,Pzu,Ppuk,U_future,U_current, support_foot_position_,jerks_x,jerks_y,x_state,y_state,Xf,Yf,SmVector);
    // For feet constraints (see Noe's C2)
    A.block(2 * N_,0, 2 * m_,4 * N_ + 2 * m_) = robot_restrictions_.GetAFootNolinealRestrictions(thetasFoot,Ppuk,Xf,Yf,step_generator_);
    
    // For orientation constraints (see Noe's C3)
    A.block(2 * N_ + 2 * m_,0,N_,4 * N_ + 2 * m_) = robot_restrictions_.GetAOrientation(Ppu,Ppuk);

    return A;
}

// Get the lower vector for inequalities
Matrix_t VisualFeatureBase::nlGetlbA()
{
    Matrix_t lbA(3 * N_ + 2 * m_,1);
    lbA.setZero();
    const Matrix_t &Pzu = mpc_model_.get_Pzu();
    const Matrix_t &Pps = mpc_model_.get_Pps();
    const Matrix_t &Pzs = mpc_model_.get_Pzs();
    const Matrix_t &Ppuk= mpc_model_.get_Ppuk();
    const Matrix_t &tV  = mpc_model_.get_ThetaV();

    Matrix_t U_future  = step_generator_.GetFuture_U_Matrix();
    Matrix_t U_current = step_generator_.GetCurrent_U_Matrix();

    const Matrix_t &x_state    = dynamic_model_.GetXStateVector();
    const Matrix_t &y_state    = dynamic_model_.GetYStateVector();
    const Matrix_t &tcom_state = dynamic_model_.GetTCOMStateVector();

    const Matrix_t &jerks_x = u_0_.block(0,0,N_,1);
    const Matrix_t &jerks_y = u_0_.block(N_ + m_,0,N_,1);
    const Matrix_t &jerks_tf= u_0_.block(2*N_+2*m_,0,N_,1);
    const Matrix_t &Xf      = u_0_.block(N_,0,m_,1);
    const Matrix_t &Yf      = u_0_.block(2 * N_ + m_,0,m_,1);

    Matrix_t thetasFoot = tV + Ppuk*jerks_tf;

    // For ZMP restrictions.
    Matrix_t Uc = step_generator_.GetCurrent_U_Matrix();
    Matrix_t Uf = step_generator_.GetFuture_U_Matrix();
    unsigned short int s1 = step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
    unsigned short int s2 = step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));
    Matrix_t SmVector(m_+1,1); SmVector.setZero();
    SmVector(0,0) = s1; SmVector(1,0) = s2; SmVector(2,0) = N_;
    Matrix_t thetas_support(m_+1,1); thetas_support.setZero();
    thetas_support(0,0) = dynamic_model_.GetTFOOTSupport();
    thetas_support(1,0) = thetasFoot(s1-1,0);
    thetas_support(2,0) = thetasFoot(s2-1,0);

    // For ZMP constraints (see Noe's C1)
    lbA.block(0,0,2 * N_,1) = robot_restrictions_.GetZMPNolinealLowerBoundaryVector(thetas_support,Pzs,Pzu,step_generator_,U_future,U_current, support_foot_position_,jerks_x,jerks_y,x_state,y_state,Xf,Yf,SmVector) ;
    // For feet constraints (see Noe's C2)
    lbA.block(2 * N_,0,2 * m_,1) = robot_restrictions_.GetFeetNolinealLowerBoundaryVector(thetasFoot,support_foot_position_,
                                                                                          step_generator_.GetSupportPhase(), step_generator_.GetFlyingFoot(),u_0_);
    // For orientation constraints (see Noe's C3)
    lbA.block(2 * N_ + 2 * m_,0,N_,1) = robot_restrictions_.GetOrientationLowerBoundaryVector(Pps,tcom_state,tV);


    return lbA;
}

// Get the upper vector for inequalities
Matrix_t VisualFeatureBase::nlGetubA()
{
    Matrix_t ubA(3 * N_ + 2 * m_,1);
    ubA.setZero();
    const Matrix_t &Pzu = mpc_model_.get_Pzu();
    const Matrix_t &Pzs = mpc_model_.get_Pzs();
    const Matrix_t &Pps = mpc_model_.get_Pps();
    const Matrix_t &Ppuk= mpc_model_.get_Ppuk();
    const Matrix_t &tV  = mpc_model_.get_ThetaV();

    Matrix_t U_future  = step_generator_.GetFuture_U_Matrix();
    Matrix_t U_current = step_generator_.GetCurrent_U_Matrix();

    const Matrix_t &x_state    = dynamic_model_.GetXStateVector();
    const Matrix_t &y_state    = dynamic_model_.GetYStateVector();
    const Matrix_t &tcom_state = dynamic_model_.GetTCOMStateVector();

    const Matrix_t &jerks_x = u_0_.block(0,0,N_,1);
    const Matrix_t &jerks_y = u_0_.block(N_ + m_,0,N_,1);
    const Matrix_t &jerks_tf= u_0_.block(2*N_+2*m_,0,N_,1);
    const Matrix_t &Xf      = u_0_.block(N_,0,m_,1);
    const Matrix_t &Yf      = u_0_.block(2 * N_ + m_,0,m_,1);

    //
    Matrix_t thetasFoot = tV + Ppuk*jerks_tf;
    // For ZMP restrictions.
    Matrix_t Uc = step_generator_.GetCurrent_U_Matrix();
    Matrix_t Uf = step_generator_.GetFuture_U_Matrix();
    unsigned short int s1 = step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
    unsigned short int s2 = step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));
    Matrix_t SmVector(m_+1,1); SmVector.setZero();
    SmVector(0,0) = s1; SmVector(1,0) = s2; SmVector(2,0) = N_;
    Matrix_t thetas_support(m_+1,1); thetas_support.setZero();
    thetas_support(0,0) = dynamic_model_.GetTFOOTSupport();
    thetas_support(1,0) = thetasFoot(s1-1,0);
    thetas_support(2,0) = thetasFoot(s2-1,0);

    // For ZMP constraints (see Noe's C1)
    ubA.block(0,0,2 * N_,1) = robot_restrictions_.GetZMPNolinealUpperBoundaryVector(thetas_support,Pzs,Pzu,step_generator_,U_future,U_current,support_foot_position_,jerks_x,jerks_y,x_state,y_state,Xf,Yf,SmVector);
    // For feet constraints (see Noe's C2)
    ubA.block(2 * N_,0,2 * m_,1) = robot_restrictions_.GetFeetNolinealUpperBoundaryVector(thetasFoot,support_foot_position_,step_generator_.GetSupportPhase(), step_generator_.GetFlyingFoot(),u_0_);
    // For orientation constraints (see Noe's C3)
    ubA.block(2 * N_ + 2 * m_,0,N_,1) = robot_restrictions_.GetOrientationUpperBoundaryVector(Pps,tcom_state,tV);

    return ubA;
}

void VisualFeatureBase::PrepareDataForNextOptimization(const Matrix_t &solution)
{

    // Non-linear case, where both the translation and rotation problems are solved simultaneously
    if (solution.rows()>2*N_+2*m_) {   
        // Important: in this function the dynamic model has been updated with the found solution
        Matrix_t solution_u0 = solution + u_0_;
        const Matrix_t &Ppu = mpc_model_.get_Ppu();
        const Matrix_t &Pps = mpc_model_.get_Pps();
        const Matrix_t &tfoot_jerks= solution_u0.block(2*N_+2*m_, 0,N_,1);
        const Matrix_t &tcom_jerks = solution_u0.block(3*N_+2*m_, 0,N_,1);
        const Matrix_t &tcom_state = dynamic_model_.GetTCOMStateVector();
        const Matrix_t &tfeet_state= dynamic_model_.GetTFOOTStateVector();
        const FPTYPE &tfeet_support= dynamic_model_.GetTFOOTSupport();

        // the initial value is updated with the previous value if it is required
        if(u_0_prev){
            SetNewU_0(solution_u0,!step_generator_.IsSameSupportFoot());
        }

        // New CoM origin is zero    
        dynamic_model_.ResetTranslationStateVector(dynamic_model_.GetTCOM_Position());
        dynamic_model_.ResetOrientationStateVector();

        // Update support foot position which is now written in the frame mk
        // This is used in the constraints
        AffineTransformation mk1_T_mk = mk_T_mk1.inverse();
        support_foot_position_ = mk1_T_mk * support_foot_position_;
        UpdateActualVisualData();

        // Update Ppuk and theta_V for the optimization of the flying foot in the next iteration.
        Matrix_t Uc = step_generator_.GetCurrent_U_Matrix();
        Matrix_t Uf = step_generator_.GetFuture_U_Matrix();
        unsigned short int s1 = step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
        unsigned short int s2 = step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));
        mpc_model_.SetThetaV(s1,s2,N_,tfeet_support,tfeet_state);
        mpc_model_.SetPpuk(s1,s2,N_);
    }
    // Linear case, where the rotations are solved first
    else {
        // New CoM origin is set to zero
        dynamic_model_.ResetTranslationStateVector(rotation_controller_->GetTrunkOrientation()(0,0));

        // Update support foot position which is now written in the frame mk
        AffineTransformation mk1_T_mk = mk_T_mk1.inverse();
        support_foot_position_ = mk1_T_mk * support_foot_position_;

        // Update the target angle to reach from the current essential matrix
        // Optimize the jerks to apply, update the rotation controller
        SolveOrientation();

        // Update the elements that will be used for each visul feature (predicted values)
        UpdateActualVisualData();
    }
    
    // Update the error components gains
    error_gains_ = GetReferenceWeights();

}