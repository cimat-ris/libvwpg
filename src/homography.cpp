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
#include <algorithm>

#include <boost/lexical_cast.hpp>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "formulations.h"
#include "vision_utils.h"

// Constructor
HomographyBase::HomographyBase(const boost::property_tree::ptree &parameters):
    VisualFeatureBase(parameters), homography_solution_(homographySolution::NONE) {
    // Initialize the names of the visual features to be used (h11, h12, h13, h31, h32, h33)
    common::logString(": initialize visual features ");
    all_visual_features_names.push_back("h11");
    all_visual_features_names.push_back("h12");
    all_visual_features_names.push_back("h13");
    all_visual_features_names.push_back("h31");
    all_visual_features_names.push_back("h32");
    all_visual_features_names.push_back("h33");

    for (auto visual_feature_name : all_visual_features_names)
        latest_visual_data_[visual_feature_name] = 0.0;

    // And their expected values
    expected_values["h11"]=1.0;
    expected_values["h12"]=0.0;
    expected_values["h13"]=0.0;
    expected_values["h31"]=0.0;
    expected_values["h32"]=0.0;
    expected_values["h33"]=1.0;
}

// Constructor to use with python
HomographyBase::HomographyBase(const std::string nameFileParameters):
    VisualFeatureBase(nameFileParameters), homography_solution_(homographySolution::NONE) {
    // Initialize the names of the visual features to be used (h11, h12, h13, h31, h32, h33)
    common::logString(": initialize visual features ");
    all_visual_features_names.push_back("h11");
    all_visual_features_names.push_back("h12");
    all_visual_features_names.push_back("h13");
    all_visual_features_names.push_back("h31");
    all_visual_features_names.push_back("h32");
    all_visual_features_names.push_back("h33");

    for (auto visual_feature_name : all_visual_features_names)
        latest_visual_data_[visual_feature_name] = 0.0;

    // And their expected values
    expected_values["h11"]=1.0;
    expected_values["h12"]=0.0;
    expected_values["h13"]=0.0;
    expected_values["h31"]=0.0;
    expected_values["h32"]=0.0;
    expected_values["h33"]=1.0;
}

// Rotation reference is set with respect to the first target image
void HomographyBase::SolveOrientation()
{
    // Get the first initial position-reference position homography
    const Matrix_t &homography_matrix = homography_matrices_[first_reference_image_];

    Matrix_t R; Vector3D_t t,n;
    FPTYPE d;
    // Decompose the first homography
    RecoverFromHomography(homography_matrix,R,t,n,d,current_iteration_,homography_solution_);

    // Deduce the angle to target
    FPTYPE target_angle = -common::YRotationAngleFromRotationMatrix(R);
    // Pass it to the rotation controller as a reference angle
    rotation_controller_->UpdateReference(target_angle);
    // Performs the optimization for the next step
    rotation_controller_->ComputeRobotNextOrientation();
}

// Compute the series of homography matrices from the current points to the reference points
void HomographyBase::SimulateVisualFeatures()
{
    // Cycle over the subsequent target images
    for (int i = first_reference_image_; i < last_reference_image_; i++) {
        // Get the corresponding 3D target image points
        InterestPoints &current_interest_points    = all_interest_points_[i];
        InterestPoints &current_interest_points_gt = all_ground_truth_interest_points_[i];
        std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];

        // Ground truth section.
        const std::vector<Point2D_t> &reference_image_points_gt = current_interest_points_gt.GetInReferenceImageCoordinates();
        const std::vector<Point2D_t> &current_image_points_gt   = current_interest_points_gt.SimulatedProjection(camera_ground_truth_);
        const std::vector<bool>        &current_visibility_gt   = current_interest_points_gt.GetVisibility();

        std::vector<cv::Point2d> cv_reference_points_gt;
        std::vector<cv::Point2d> cv_current_points_gt;
        OpencvVector(reference_image_points_gt,current_image_points_gt,current_visibility_gt,cv_reference_points_gt,cv_current_points_gt);

        Matrix_t homography_matrix_gt = ComputeHomography(cv_current_points_gt,cv_reference_points_gt,camera_ground_truth_.GetIntrisicParametersMatrix());
        homography_matrix_gt = (1.0 / homography_matrix_gt(1, 1)) * homography_matrix_gt;
        homography_matrices_ground_truth_[i] = homography_matrix_gt;

        ///Simulated section
        // Reference points in image coordinates
        const std::vector<Point2D_t> &reference_image_points = current_interest_points.GetInReferenceImageCoordinates();
        const std::vector<Point2D_t> &current_image_points   = current_interest_points.SimulatedProjection(camera_ground_truth_,true);
        const std::vector<bool>        &current_visibility   = current_interest_points.GetVisibility();


        if (i == first_reference_image_) {
            const std::vector<Point3D_t> &reference_world_points = current_interest_points.GetInCurrentWorldCoordinates();  
//            for (int k=0;k<current_image_points.size();k++) {
//                common::Log3DInformation("[current_world_point]", current_iteration_, i, k,  reference_world_points[k]);
//                common::Log2DInformation("[current_image_point]", current_iteration_, i, k, current_image_points[k], current_visibility[k]);
//            }
        }

        std::vector<cv::Point2d> cv_reference_points;
        std::vector<cv::Point2d> cv_current_points;
        OpencvVector(reference_image_points_gt,current_image_points,current_visibility,cv_reference_points,cv_current_points);
        Matrix_t homography_matrix = Matrix_t::Identity(3,3);

        if(cv_reference_points.size()>min_points_to_estimate_constraint_) {// Computes the homography from current to reference image points if there are enough points
            homography_matrix = ComputeHomography(cv_current_points, cv_reference_points,camera_.GetIntrisicParametersMatrix());
            // OpenCV normalizes the Homography matrix with respect to h33. For planar motion
            // h22 = 1, thus the homography matrix is re-normalized with respect to h22
            homography_matrix = (1.0 / homography_matrix(1, 1)) * homography_matrix;
        }
        else{
            // When not enough features are available, use the current prediction
            for (VisualFeatureData &visual_feature : current_visual_features) {
                std::pair<int,int> p = visual_feature.model->GetMatrixPosition();
                // Actual and predicted values of the visual feature
                homography_matrix(p.first, p.second) = visual_feature.predicted;
            }
        }
        homography_matrices_[i] = homography_matrix;
    }
}

// Compute the matrix H (for the solver)
Matrix_t HomographyBase::GetH()
{
    Matrix_t H(2*(N_+m_), 2*(N_+m_));
    Matrix_t Q_prime(N_+m_, N_+m_);
    Matrix_t Q_hat(2*(N_+m_), 2*(N_+m_));

    const Matrix_t &Pvu = mpc_model_.get_Pvu();
    const Matrix_t &Pzu = mpc_model_.get_Pzu();
    const Matrix_t &Ppu = mpc_model_.get_Ppu();
    const Matrix_t &Pau = mpc_model_.get_Pau();
    Matrix_t U_future = step_generator_.GetFuture_U_Matrix();

    Q_prime.setZero();
    Q_prime.block(0, 0, N_, N_) = alpha_*Matrix_t::Identity(N_, N_) + eta_x_ * Pvu.transpose() * Pvu +
                                  gamma_ * Pzu.transpose() * Pzu + kappa_ * Pau.transpose() * Pau;
    Q_prime.block(0, N_, N_, m_) = -gamma_*Pzu.transpose()*U_future;
    Q_prime.block(N_, 0, m_, N_) = -gamma_*U_future.transpose()*Pzu;
    Q_prime.block(N_, N_, m_, m_) = gamma_*U_future.transpose()*U_future;

    Q_hat.setZero();

    switch(multiple_objective_method_){
        case MultipleObjectives::WeightedAverages :
            // Cycle over the reference images
            for (int i = first_reference_image_; i < last_reference_image_ && i - first_reference_image_ < REFERENCE_WINDOW_; i++) {
                std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];
                FPTYPE error_gain = error_gains_[i - first_reference_image_];

                for (VisualFeatureData &visual_feature : current_visual_features) {
                    MPCLinearHomography *hom_mpc = dynamic_cast<MPCLinearHomography *>(visual_feature.model.get());
                    if (!hom_mpc)
                        throw std::runtime_error("ERROR: the pointer is not MPCLinearHomography*");
                    Matrix_t A = hom_mpc->A();
                    Matrix_t B = hom_mpc->B();
                    Matrix_t C = hom_mpc->C();
                    const FPTYPE &beta = visual_feature.model->GetGain();

                    Q_hat.block(0, 0, N_, N_) += error_gain * beta*Ppu.transpose() * A.transpose() * A * Ppu;
                    Q_hat.block(0, N_ + m_, N_, N_) += error_gain * beta*Ppu.transpose() * A.transpose() * B * Ppu;
                    Q_hat.block(N_ + m_, 0 , N_, N_) += error_gain * beta*Ppu.transpose() * B.transpose() * A * Ppu;
                    Q_hat.block(N_ + m_, N_ + m_, N_, N_) += error_gain * beta*Ppu.transpose() * B.transpose() * B * Ppu;
                }
            }
            break;
        case MultipleObjectives::SharedPredictionWindows :
            // Cycle over the reference images
            for (int i = first_reference_image_; i < last_reference_image_ && i - first_reference_image_ < REFERENCE_WINDOW_; i++) {
                std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];
                std::vector<VisualFeatureData> &current_visual_features_Ds = all_visual_features_[first_reference_image_];
                Matrix_t W;
                if((i==0)&& (current_iteration_==0)){
                    W  = Matrix_t::Identity(N_,N_);
                }
                else if ((i==1)&& (current_iteration_==0))
                    W = Matrix_t::Zero(N_,N_);
                else if(i==first_reference_image_ && current_iteration_!=0 && (first_reference_image_+1-last_reference_image_ !=0)){
                    Matrix_t Ds = GetDs(current_visual_features_Ds);
                    W  = Ds.transpose()*Ds;
                }
                else if(i==first_reference_image_+1 && current_iteration_!=0 && (first_reference_image_+1-last_reference_image_ !=0) ){
                    Matrix_t I = Matrix_t::Identity(N_,N_);
                    Matrix_t Ds = GetDs(current_visual_features_Ds);
                    Matrix_t IminDs = I - Ds;
                    W = IminDs.transpose()*IminDs;
                }else
                    W  = Matrix_t::Identity(N_,N_);

                for (VisualFeatureData &visual_feature : current_visual_features) {
                    MPCLinearHomography *hom_mpc = dynamic_cast<MPCLinearHomography *>(visual_feature.model.get());
                    if (!hom_mpc)
                        throw std::runtime_error("ERROR: the pointer is not MPyaCLinearHomography*");
                    Matrix_t A = hom_mpc->A();
                    Matrix_t B = hom_mpc->B();
                    Matrix_t C = hom_mpc->C();
                    const FPTYPE &beta = visual_feature.model->GetGain();

                    Q_hat.block(0, 0, N_, N_)             +=  beta*Ppu.transpose() * A.transpose() * W * A * Ppu;
                    Q_hat.block(0, N_ + m_, N_, N_)       +=  beta*Ppu.transpose() * A.transpose() * W * B * Ppu;
                    Q_hat.block(N_ + m_, 0 , N_, N_)      +=  beta*Ppu.transpose() * B.transpose() * W * A * Ppu;
                    Q_hat.block(N_ + m_, N_ + m_, N_, N_) +=  beta*Ppu.transpose() * B.transpose() * W * B * Ppu;
                }
            }
            break;
        default:
            break;
    }


    H.setZero();
    H.block(0, 0, N_+m_, N_+m_) = Q_prime;
    Q_prime.block(0, 0, N_, N_) = alpha_*Matrix_t::Identity(N_, N_) + eta_y_ * Pvu.transpose() * Pvu +
                                  gamma_ * Pzu.transpose() * Pzu + kappa_ * Pau.transpose() * Pau;
    H.block(N_+m_, N_+m_, N_+m_, N_+m_) = Q_prime;

    return H + Q_hat;
}

// Compute the vector g (for the solver)
Matrix_t HomographyBase::Getg()
{
    Matrix_t g(2*N_ + 2*m_, 1);
    Matrix_t p_hat(2*N_ + 2*m_, 1);

    const Matrix_t &Pvu = mpc_model_.get_Pvu();
    const Matrix_t &Pzu = mpc_model_.get_Pzu();
    const Matrix_t &Ppu = mpc_model_.get_Ppu();
    const Matrix_t &Pzs = mpc_model_.get_Pzs();
    const Matrix_t &Pvs = mpc_model_.get_Pvs();
    const Matrix_t &Pps = mpc_model_.get_Pps();
    const Matrix_t &Pau = mpc_model_.get_Pau();
    const Matrix_t &Pas = mpc_model_.get_Pas();

    Matrix_t U_future = step_generator_.GetFuture_U_Matrix();
    Matrix_t U_current = step_generator_.GetCurrent_U_Matrix();

    Matrix_t x_state = dynamic_model_.GetXStateVector();
    Matrix_t y_state = dynamic_model_.GetYStateVector();

    g.setZero();
    g.block(0, 0, N_, 1) = eta_x_ * Pvu.transpose() * (Pvs * x_state - Matrix_t::Constant(N_, 1, x_speed_ref_) ) +
                           gamma_*Pzu.transpose() * (Pzs*x_state - U_current * X(support_foot_position_)) +
                           kappa_ * Pau.transpose() * Pas * x_state;
    g.block(N_, 0, m_, 1) = -gamma_*U_future.transpose()*(Pzs*x_state - U_current * X(support_foot_position_));

    g.block(N_+m_, 0, N_, 1) = eta_y_ * Pvu.transpose() * (Pvs * y_state  )+
                               gamma_ * Pzu.transpose() * (Pzs*y_state - U_current*Y(support_foot_position_)) +
                               kappa_ * Pau.transpose() * Pas * y_state;
    g.block(N_+m_+N_, 0, m_, 1) = -gamma_*U_future.transpose()*(Pzs*y_state - U_current*Y(support_foot_position_));

    p_hat.setZero();

    FPTYPE objectiveFunctionValue = 0.0;
    switch(multiple_objective_method_){
        case MultipleObjectives::WeightedAverages :
            // Cycle over the reference images
            for (int i = first_reference_image_; i < last_reference_image_ && i - first_reference_image_ < REFERENCE_WINDOW_; i++) {
                std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];
                const FPTYPE &error_gain = error_gains_[i - first_reference_image_];

                for (VisualFeatureData &visual_feature : current_visual_features) {
                    MPCLinearHomography *hom_mpc = dynamic_cast<MPCLinearHomography *>(visual_feature.model.get());
                    if (!hom_mpc)
                        throw std::runtime_error("ERROR: the pointer is not MPCLinearHomography*");
                    Matrix_t A = hom_mpc->A();
                    Matrix_t B = hom_mpc->B();
                    Matrix_t C = hom_mpc->C();
                    const FPTYPE &beta = hom_mpc->GetGain();
                    const FPTYPE &actual    = visual_feature.actual;
                    const FPTYPE &predicted = visual_feature.predicted;
                    FPTYPE desired_value = visual_feature.model->GetExpectedValue() - (actual - predicted);
                    Matrix_t common = A * Pps * x_state + B * Pps * y_state + C - Matrix_t::Constant(N_, 1, desired_value);
                    p_hat.block(0, 0, N_, 1)       += error_gain * beta*Ppu.transpose() * A.transpose() * common;
                    p_hat.block(N_ + m_, 0, N_, 1) += error_gain * beta*Ppu.transpose() * B.transpose() * common;
                    if(current_iteration_ != 0){
                        const FPTYPE &beta = visual_feature.model->GetGain();
                        Matrix_t Hres = hom_mpc->GetHp() - Matrix_t::Constant(N_, 1, desired_value);
                        Matrix_t res = beta*Hres.transpose()*Hres;
                        objectiveFunctionValue += res(0,0);
                    }
                    }
                 common::logString("[objective_function]: " + boost::lexical_cast<std::string>(objectiveFunctionValue));
            }
            break;
        case MultipleObjectives::SharedPredictionWindows:
            // Cycle over the reference images
            for (int i = first_reference_image_; i < last_reference_image_ && i - first_reference_image_ < REFERENCE_WINDOW_; i++) {
                std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];
                std::vector<VisualFeatureData> &current_visual_features_Ds = all_visual_features_[first_reference_image_];
                Matrix_t W;
                if( ( (i==0)&& (current_iteration_==0) )  ){
                    W  = Matrix_t::Identity(N_,N_);
                }
                else if ((i==1)&& (current_iteration_==0))
                    W = Matrix_t::Zero(N_,N_);
                else if(i==first_reference_image_ && current_iteration_!=0 && (first_reference_image_+1-last_reference_image_ !=0)){
                    Matrix_t Ds = GetDs(current_visual_features_Ds);
                    W  = Ds.transpose()*Ds;
                }
                else if(i==first_reference_image_+1 && current_iteration_!=0 && (first_reference_image_+1-last_reference_image_ !=0) ){
                    Matrix_t I = Matrix_t::Identity(N_,N_);
                    Matrix_t Ds = GetDs(current_visual_features_Ds);
                    Matrix_t IminDs = I - Ds;
                    W = IminDs.transpose()*IminDs;
                } else
                    W  = Matrix_t::Identity(N_,N_);


                for (VisualFeatureData &visual_feature : current_visual_features) {
                    MPCLinearHomography *hom_mpc = dynamic_cast<MPCLinearHomography *>(visual_feature.model.get());
                    if (!hom_mpc)
                        throw std::runtime_error("ERROR: the pointer is not MPCLinearHomography*");
                    Matrix_t A = hom_mpc->A();
                    Matrix_t B = hom_mpc->B();
                    Matrix_t C = hom_mpc->C();
                    const FPTYPE &beta = hom_mpc->GetGain();
                    const FPTYPE &actual    = visual_feature.actual;
                    const FPTYPE &predicted = visual_feature.predicted;
                    FPTYPE desired_value = visual_feature.model->GetExpectedValue() - (actual - predicted);
                    Matrix_t common = A * Pps * x_state + B * Pps * y_state + C - Matrix_t::Constant(N_, 1, desired_value);
                    p_hat.block(0, 0, N_, 1)       +=  beta*Ppu.transpose() * A.transpose() * W * common;
                    p_hat.block(N_ + m_, 0, N_, 1) +=  beta*Ppu.transpose() * B.transpose() * W * common;
                }
            }

            break;
        default:
            break;
    }

    return g + p_hat;
}

Matrix_t HomographyBase::GetDs(std::vector<VisualFeatureData> &current_visual_features) {
    std::vector<int > indices;
    indices.clear();
    int index;
    int value_index = 0;
    int counter = 0;
    for (VisualFeatureData &visual_feature : current_visual_features) {
        MPCLinearHomography *hom_mpc = dynamic_cast<MPCLinearHomography *>(visual_feature.model.get());
        if (!hom_mpc)
            throw std::runtime_error("ERROR: the pointer is not MPCLinearHomography*");
        Matrix_t Hp = hom_mpc->GetHp();
        index = GetIndexl(visual_feature.model->GetExpectedValue(),Hp);
        indices.push_back(index);
        if(index<N_)
            counter++;

        if(index<6)
            value_index++;
    }

    int min_index_;
    if(counter==6){
        auto min_index = std::min_element(indices.begin(),indices.end());
        min_index_ = *min_index;
    }
    else{
        min_index_ = N_;
    }


    Matrix_t Ds(N_,N_); Ds.setZero();
    if((min_index_)<=N_){
        for(int i=0;i<(min_index_);i++)
            Ds(i,i) = 1;
    }

    if(value_index>5)
        index_switch_threshold_ = true;
    else
        index_switch_threshold_ = false;

    return Ds;
}

FPTYPE HomographyBase::DistanceFromPlane(const Vector3D_t &n, FPTYPE dInit, FPTYPE xk, FPTYPE yk)
{
    Eigen::Vector3d c(-yk,0.0,xk);
    FPTYPE d     = dInit - n.adjoint()*c;
    return (1.0/d);
}

// Update the visual features from the current frame info
// This code supposes that:
// - the homography matrices have been updated before.
// - the rotation controller has been run in the linear case.
// - the predicted values from the last step have been calculated before.
void HomographyBase::UpdateActualVisualData()
{
	// For all the reference images
 	for (int i = first_reference_image_; i < last_reference_image_; i++) {
 		// Get the homography and decompose it
    	const Matrix_t &homography_matrix    = homography_matrices_[i];
    	const Matrix_t &homography_matrix_gt = homography_matrices_ground_truth_[i];
    	Matrix_t R;
    	Vector3D_t t;
    	Vector3D_t n;
    	FPTYPE d;
    	RecoverFromHomography(homography_matrix,R,t,n,d,current_iteration_,homography_solution_);
    	FPTYPE phi = common::YRotationAngleFromRotationMatrix(R);
        std::ostringstream os;
        os << "[visual_measures]: "
           << "iteration=" << current_iteration_
           << ", reference_id=" << std::setw(3) << std::setfill('0') << i
           << ", phi=" << phi
           << ", t=" << t.transpose();
        common::logString(os.str());

    	// Get the visual feature data
    	std::vector<VisualFeatureData> &visual_data_for_current_image = all_visual_features_[i];
    	for (VisualFeatureData& visual_feature : visual_data_for_current_image) {
            visual_feature.model->ck_x_t(X(t));
            visual_feature.model->ck_z_t(Z(t));
            visual_feature.model->phi(phi);
            visual_feature.model->c_x_com(c_x_com_);
            visual_feature.model->c_z_com(c_z_com_);
            // TODO: do this with polymorphism?
            MPCLinearHomography *hom_mpc = dynamic_cast<MPCLinearHomography *>(visual_feature.model.get());
            if (hom_mpc){
                // Update the parameters for the next prediction
                hom_mpc->thetas(rotation_controller_->GetTrunkOrientation());
                // Update the ck_x_t, ck_z_t parameters, which are the target position in the current camera frame
                // with z pointing onwards, x on the right. They are easily deduced from the essential matrix.
                hom_mpc->d(d_);
                hom_mpc->nx(X(n));
                hom_mpc->ny(Y(n));
                hom_mpc->nz(Z(n));
            }
            MPCNonLinearHomography *hom_nlmpc = dynamic_cast<MPCNonLinearHomography *>(visual_feature.model.get());
            if (hom_nlmpc) {
                // Update the ck_x_t, ck_z_t parameters, which are the target position in the current camera frame
                // with z pointing onwards, x on the right. They are easily deduced from the essential matrix.
                hom_nlmpc->d(d_);
                hom_nlmpc->nx(X(n));
                hom_nlmpc->ny(Y(n));
                hom_nlmpc->nz(Z(n));
            }
            // A visual feature is associated to position in the homography matrix
            std::pair<int,int> p = visual_feature.model->GetMatrixPosition();
            // Actual and predicted values of the visual feature
            visual_feature.actual = homography_matrix(p.first, p.second);
            // Predicted value of the visual feature
            if (i == first_reference_image_) {
                   std::ostringstream oss;
                   oss << "[visual_feature]: "
                     << "iteration=" << current_iteration_
                     << ", name=" << visual_feature.model->GetName()
                     << ", actual=" << visual_feature.actual
                     << ", predicted=" << visual_feature.predicted
                     << ", expected=" << visual_feature.model->GetExpectedValue();
                      // << ", ground_truth=" << homography_matrix_gt(p.first, p.second);
                   common::logString(oss.str());
                    // only keep track of visual data for the first target
                    FPTYPE& current_visual_average = latest_visual_data_[visual_feature.model->GetName()];
                    if (number_of_samples_ > 20)
                        current_visual_average -= current_visual_average / 20;
                    current_visual_average += visual_feature.actual / 20;
            }
    	}

    	if (i == first_reference_image_) {
            // Some stuff that is done only for the first reference image
            RefAnglesInterpolation(-phi);
      		++number_of_samples_;
        }
  	}
}

void HomographyBase::UpdatePredictedValues(const Matrix_t &solution)
{
    FPTYPE xx,yy,tt;
    const Matrix_t &Ppu = mpc_model_.get_Ppu();
    const Matrix_t &Pps = mpc_model_.get_Pps();
    const Matrix_t &xcom_state = dynamic_model_.GetXStateVector();
    const Matrix_t &ycom_state = dynamic_model_.GetYStateVector();
    Matrix_t Xk1;
    Matrix_t Yk1;
    if (solution.rows()) {
        xx  = dynamic_model_.GetCOM_X_NextPosition(solution(0,0));
        yy  = dynamic_model_.GetCOM_Y_NextPosition(solution(N_+  m_,0));
        Xk1 = Pps*xcom_state + Ppu*solution.block(  0      , 0, N_, 1);
        Yk1 = Pps*ycom_state + Ppu*solution.block(  N_+  m_, 0, N_, 1);
        if (solution.rows()==2*N_+2*m_) {
            // Linear case (orientations are optimized in a different object)
            tt     = rotation_controller_->GetTrunkOrientation()(0,0);
        } else {
            // Non-linear case (orientations are optimized simultaneously)
            // This is the trunk orientation, with an integrator
            tt = dynamic_model_.GetTCOM_NextPosition(solution(3*N_+2*m_,0));
        }
    }
    // For all the target images
    for (int i = first_reference_image_; i < last_reference_image_; i++) {
        // For all the visual feature data
        std::vector<VisualFeatureData> &visual_data_for_current_image = all_visual_features_[i];
        for (VisualFeatureData& visual_feature : visual_data_for_current_image) {
            // Set the predicted value of the visual feature
            if (!solution.rows())
                // The first time, we simply take 0.0
                visual_feature.predicted = 0.0;
            else{
                visual_feature.predicted = visual_feature.model->Predicted(xx,yy,tt);
            }
            MPCLinearHomography *hom_mpc = dynamic_cast<MPCLinearHomography *>(visual_feature.model.get());
            if (hom_mpc){
                Matrix_t A  = hom_mpc->A();
                Matrix_t B  = hom_mpc->B();
                Matrix_t C  = hom_mpc->C();
                Matrix_t Hp = A*Xk1 + B*Yk1 + C;
                hom_mpc->SetHp(Hp);
            }
        }
    }

    if (logPredictions_)
        LogCurrentPredictions(solution);
}

// Constructor: Simulated homography (used 3D points)
HomographySimulated::HomographySimulated(const boost::property_tree::ptree &parameters):
	HomographyBase(parameters)
{

    // World-to-com transformation
    auto com_T_world = world_T_com_.inverse();

    // Initialize reference data. This cycles over target images
    while (true) {
        std::vector<Point3D_t> reference_points_in_world_coordinates;
    	// For each target image, load 3d reference points in world coordinates from the ini file or parameters to generate them
        try {
            reference_points_in_world_coordinates = Load3DReferencePoints_(parameters, com_T_world, last_reference_image_);
        } catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ptree_bad_path> >) {
            break;
        } 
        // Generate the reference points in image coordinates (all_interest_points_ is formed/completed there)
        InitReferenceImage_(parameters, reference_points_in_world_coordinates, last_reference_image_, false, camera_ground_truth_,false);
        
        // Generate the reference points in image coordinates for the ground truth (all_ground_truth_interest_points_ is formed/completed there)
        InitReferenceImage_(parameters, reference_points_in_world_coordinates, last_reference_image_, false, camera_ground_truth_,true);
        ++last_reference_image_;
    }
	
	// This is the number of target images
    std::string log_entry(": found " + boost::lexical_cast<std::string>(last_reference_image_) + " reference point(s) in configuration file");
    common::logString(log_entry);

	 // Initializes as many homographies as target images
    homography_matrices_              = std::vector<Matrix_t>(last_reference_image_);
    homography_matrices_ground_truth_ = std::vector<Matrix_t>(last_reference_image_);

    // Initialize (homography) visual features objects with their model parameters
    // The all_interest_points_ has been initialized by InitReferenceImage_
    common::logString(": initialize interest points ");
    for (InterestPoints points : all_interest_points_) {
        std::vector<VisualFeatureData> visual_model_for_reference;
        bool linear = true;
        // Check if we will use a linear/non-linear solver
        try {
            if (parameters.get<std::string>("qp.linear")!="true")
                linear=false;
        } catch (boost::property_tree::ptree_bad_path e) {
            // Case the linear flag has not been found. Consider this as the linear case.
        }
        if (linear) {
            common::logString(": using a linear formulation of homography matrix entries ");
            // For all visual features (h11,h33...)
            for (auto visual_feature_name : all_visual_features_names ) {
                visual_model_for_reference.push_back(VisualFeatureData(visual_feature_name, N_));
                std::string gain_name = std::string("qp.beta") + visual_feature_name;
                visual_model_for_reference.back().model->gain(parameters.get<FPTYPE>(gain_name));
            }
        } else {
            common::logString(": using a non-linear formulation of homography matrix entries ");
            // For all visual features (h11,h33...)
            for (auto visual_feature_name : all_visual_features_names ) {
                visual_model_for_reference.push_back(VisualFeatureData(visual_feature_name, N_,false));
                std::string gain_name = std::string("qp.beta") + visual_feature_name;
                visual_model_for_reference.back().model->gain(parameters.get<FPTYPE>(gain_name));
            }
        }
        all_visual_features_.push_back(visual_model_for_reference);
    }

    // Reference weigths
    common::logString(": getting reference weights");
    error_gains_ = GetReferenceWeights();

    // Compute the series of homography matrices from the current points to the reference points
    common::logString(": compute homography matrices ");
    SimulateVisualFeatures();

    if (parameters.get<std::string>("qp.linear")=="true"){
        // Solve the first rotation sequence. The reference is set with respect to the first target image.
        common::logString(": update rotation reference ");
        SolveOrientation();
    }

    // Update the visual features (actual and predicted) from current frame
    common::logString(": update visual data ");
    UpdateActualVisualData();

    // Log the image positions of the reference points in the current position
    TakePictureReferenceFromCurrentPosition();
}

// Constructor: Simulated homography (used 3D points) to use with python
HomographySimulated::HomographySimulated(const std::string nameFileParameters):
    HomographyBase(nameFileParameters)
{
    // World-to-com transformation
    auto com_T_world = world_T_com_.inverse();

    // Initialize reference data. This cycles over target images
    while (true) {
        std::vector<Point3D_t> reference_points_in_world_coordinates;
        // For each target image, load 3d reference points in world coordinates from the ini file or parameters to generate them
        try {
            reference_points_in_world_coordinates = Load3DReferencePoints_(parameters_, com_T_world, last_reference_image_);
        } catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ptree_bad_path> >) {
            break;
        } 
        // Generate the reference points in image coordinates (all_interest_points_ is formed/completed there)
        InitReferenceImage_(parameters_, reference_points_in_world_coordinates, last_reference_image_, false, camera_ground_truth_,false);
        
        // Generate the reference points in image coordinates for the ground truth (all_ground_truth_interest_points_ is formed/completed there)
        InitReferenceImage_(parameters_, reference_points_in_world_coordinates, last_reference_image_, false, camera_ground_truth_,true);
        ++last_reference_image_;
    }
    
    // This is the number of target images
    std::string log_entry(": found " + boost::lexical_cast<std::string>(last_reference_image_) + " reference point(s) in configuration file");
    common::logString(log_entry);

     // Initializes as many homographies as target images
    homography_matrices_              = std::vector<Matrix_t>(last_reference_image_);
    homography_matrices_ground_truth_ = std::vector<Matrix_t>(last_reference_image_);

    // Initialize (homography) visual features objects with their model parameters
    // The all_interest_points_ has been initialized by InitReferenceImage_
    common::logString(": initialize interest points ");
    for (InterestPoints points : all_interest_points_) {
        std::vector<VisualFeatureData> visual_model_for_reference;
        bool linear = true;
        // Check if we will use a linear/non-linear solver
        try {
            if (parameters_.get<std::string>("qp.linear")!="true")
                linear=false;
        } catch (boost::property_tree::ptree_bad_path e) {
            // Case the linear flag has not been found. Consider this as the linear case.
        }
        if (linear) {
            common::logString(": using a linear formulation of homography matrix entries ");
            // For all visual features (h11,h33...)
            for (auto visual_feature_name : all_visual_features_names ) {
                visual_model_for_reference.push_back(VisualFeatureData(visual_feature_name, N_));
                std::string gain_name = std::string("qp.beta") + visual_feature_name;
                visual_model_for_reference.back().model->gain(parameters_.get<FPTYPE>(gain_name));
            }
        } else {
            common::logString(": using a non-linear formulation of homography matrix entries ");
            // For all visual features (h11,h33...)
            for (auto visual_feature_name : all_visual_features_names ) {
                visual_model_for_reference.push_back(VisualFeatureData(visual_feature_name, N_,false));
                std::string gain_name = std::string("qp.beta") + visual_feature_name;
                visual_model_for_reference.back().model->gain(parameters_.get<FPTYPE>(gain_name));
            }
        }
        all_visual_features_.push_back(visual_model_for_reference);
    }

    // Reference weigths
    common::logString(": getting reference weights");
    error_gains_ = GetReferenceWeights();

    // Compute the series of homography matrices from the current points to the reference points
    common::logString(": compute homography matrices ");
    SimulateVisualFeatures();

    if (parameters_.get<std::string>("qp.linear")=="true"){
        // Solve the first rotation sequence. The reference is set with respect to the first target image.
        common::logString(": update rotation reference ");
        SolveOrientation();
    }

    // Update the visual features (actual and predicted) from current frame
    common::logString(": update visual data ");
    UpdateActualVisualData();

    // Log the image positions of the reference points in the current position
    TakePictureReferenceFromCurrentPosition();
}


// Constructor
Homography::Homography(const boost::property_tree::ptree &parameters,
                       const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                       const FPTYPE h21, const FPTYPE h22, const FPTYPE h23,
                       const FPTYPE h31, const FPTYPE h32, const FPTYPE h33 ):
    HomographyBase(parameters)
{

    common::logDesiredPosition(current_iteration_,parameters.get<FPTYPE>("reference.camera_position0_x"), parameters.get<FPTYPE>("reference.camera_position0_y"),
                           parameters.get<FPTYPE>("reference.orientation0" ) );
    Matrix_t homography_matrix(3, 3);
    homography_matrix << h11, h12, h13, h21, h22, h23, h31, h32, h33;
    last_reference_image_=1;
    // This is the number of target images
    std::string log_entry(": found " + boost::lexical_cast<std::string>(last_reference_image_) + " reference point(s) in configuration file");
    common::logString(log_entry);

    // As many Homographies as target images
    homography_matrices_ = std::vector<Matrix_t>(last_reference_image_);
    homography_matrices_[first_reference_image_] = (1.0 / homography_matrix(1, 1)) * homography_matrix;

    // Initialize (homography) visual features objects
    common::logString(": initializes visual data");
    for (Matrix_t matrices : homography_matrices_) { 
      std::vector<VisualFeatureData> visual_model_for_reference;
      // For all visual features (h11,h13...)
      for (auto visual_feature_name : all_visual_features_names ) {
        visual_model_for_reference.push_back(VisualFeatureData(visual_feature_name, N_));
        std::string gain_name = std::string("qp.beta") + visual_feature_name;
        visual_model_for_reference.back().model->gain(parameters.get<FPTYPE>(gain_name));
      }
      all_visual_features_.push_back(visual_model_for_reference);
    }

    // Reference weigths
    common::logString(": getting reference weights");
    error_gains_ = GetReferenceWeights();


    if (parameters.get<std::string>("qp.linear")=="true"){
        // Solve the first rotation sequence. The reference is set with respect to the first target image.
        common::logString(": initializes orientation ");
        SolveOrientation();
    }

    common::logString(": update visual data ");
    // Update the visual features (actual and predicted) from current frame
    UpdateActualVisualData();


    common::logString(": homography constructor OK");
}

Homography::Homography(const std::string nameFileParameters,
           const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
           const FPTYPE h21, const FPTYPE h22, const FPTYPE h23,
           const FPTYPE h31, const FPTYPE h32, const FPTYPE h33 ):
           HomographyBase(nameFileParameters){

    common::logString(": init homography constructor ");
    common::logDesiredPosition(current_iteration_,parameters_.get<FPTYPE>("reference.camera_position0_x"), parameters_.get<FPTYPE>("reference.camera_position0_y"),
               parameters_.get<FPTYPE>("reference.orientation0" ) );

    Matrix_t homography_matrix(3, 3);

    homography_matrix << h11, h12, h13, h21, h22, h23, h31, h32, h33;
    last_reference_image_=1;
    // This is the number of target images
    std::string log_entry(": found " + boost::lexical_cast<std::string>(last_reference_image_) + " reference point(s) in configuration file");
    common::logString(log_entry);

    // As many Homographies as target images
    homography_matrices_ = std::vector<Matrix_t>(last_reference_image_);
    homography_matrices_[first_reference_image_] = (1.0 / homography_matrix(1, 1)) * homography_matrix;

    // Initialize (homography) visual features objects
    common::logString(": initializes visual data");
    for (Matrix_t matrices : homography_matrices_) {
        std::vector<VisualFeatureData> visual_model_for_reference;
        // For all visual features (h11,h13...)
        for (auto visual_feature_name : all_visual_features_names ) {
            visual_model_for_reference.push_back(VisualFeatureData(visual_feature_name, N_));
            std::string gain_name = std::string("qp.beta") + visual_feature_name;
            visual_model_for_reference.back().model->gain(parameters_.get<FPTYPE>(gain_name));
        }
        all_visual_features_.push_back(visual_model_for_reference);
    }

    // Reference weigths
    common::logString(": getting reference weights");
    error_gains_ = GetReferenceWeights();


    if (parameters_.get<std::string>("qp.linear")=="true"){
        // Solve the first rotation sequence. The reference is set with respect to the first target image.
        common::logString(": initializes orientation ");
        SolveOrientation();
    }

    common::logString(": update visual data ");
    // Update the visual features (actual and predicted) from current frame
    UpdateActualVisualData();


    common::logString(": homography constructor OK");
}

void Homography::SetCurrentHomography(const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                                      const FPTYPE h21, const FPTYPE h22, const FPTYPE h23,
                                      const FPTYPE h31, const FPTYPE h32, const FPTYPE h33, bool isComputeHomography) {
    Matrix_t homography_matrix(3, 3);


    if(isComputeHomography) {// IF IS POSIBLE TO COMPUTE A HOMOGRAPHY MATRIX
        homography_matrix << h11, h12, h13, h21, h22, h23, h31, h32, h33;
        homography_matrix = (1.0 / homography_matrix(1, 1)) * homography_matrix;
    }
    else{
        std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[first_reference_image_];
        // When not enough features are available, use the current prediction
        for (VisualFeatureData &visual_feature : current_visual_features) {
            std::pair<int,int> p = visual_feature.model->GetMatrixPosition();
            // Actual and predicted values of the visual feature
            homography_matrix(p.first, p.second) = visual_feature.predicted;
        }
    }
    homography_matrices_[first_reference_image_] = homography_matrix;
}

void Homography::UpdateSimulation(const Matrix_t &solution) {
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
    world_flying_foot_orientation_ = world_com_orientation_ + footorientation;

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

    ++current_iteration_;
}

HomographyObstacles::HomographyObstacles(const boost::property_tree::ptree &parameters,
                                        const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                                        const FPTYPE h21, const FPTYPE h23,
                                        const FPTYPE h31, const FPTYPE h32, const FPTYPE h33, 
                                        const FPTYPE c1,  const FPTYPE c2,  const FPTYPE c3, const bool isObstacle):
                                        Homography(parameters,h11,h12,h13,h21,1.0,h23,h31,h32,h33)
{

    SetC1AndC2(c1,c2,0,isObstacle);
    SetBobs(c3,0,isObstacle);
    
}

HomographyObstacles::HomographyObstacles(const std::string nameFileParameters,
                                        const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                                        const FPTYPE h21, const FPTYPE h23,
                                        const FPTYPE h31, const FPTYPE h32, const FPTYPE h33, 
                                        const FPTYPE c1,  const FPTYPE c2,  const FPTYPE c3, const bool isObstacle):
                                        Homography(nameFileParameters,h11,h12,h13,h21,1.0,h23,h31,h32,h33)
{
    SetC1AndC2(c1,c2,0,isObstacle);
    SetBobs(c3,0,isObstacle);
    common::logString(": HomographyObstacles constructor OK");
}

void HomographyObstacles::SetCurrentHomography( const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                                                const FPTYPE h21, const FPTYPE h23,
                                                const FPTYPE h31, const FPTYPE h32, const FPTYPE h33, const bool isObstacle, const bool isComputeHomography,
                                                const FPTYPE c1,  const FPTYPE c2,  const FPTYPE c3, const int iteration){

    Matrix_t homography_matrix(3, 3);


    if(isComputeHomography) {// IF IS POSIBLE TO COMPUTE A HOMOGRAPHY MATRIX
        homography_matrix << h11, h12, h13, h21, 1.0, h23, h31, h32, h33;
    }
    else{
        std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[first_reference_image_];
        // When not enough features are available, use the current prediction
        for (VisualFeatureData &visual_feature : current_visual_features) {
            std::pair<int,int> p = visual_feature.model->GetMatrixPosition();
            // Actual and predicted values of the visual feature
            homography_matrix(p.first, p.second) = visual_feature.predicted;
        }
    }
    homography_matrices_[first_reference_image_] = homography_matrix;   


    SetC1AndC2(c1,c2,0,isObstacle);
    SetBobs(c3,0,isObstacle);

}

void HomographyObstacles::SetC1AndC2(const FPTYPE c1,  const FPTYPE c2, const int iteration, const bool isObstacle){
    Matrix_t C1(N_, N_);
    C1.setZero();
    Matrix_t C2(N_, N_);
    C2.setZero();

    if(isObstacle){
        for(int i=0; i<N_; i++){
            C1(i, i) = c1;
            C2(i, i) = c2;
        }
    }
    C1_ = C1;
    C2_ = C2;

}

Matrix_t HomographyObstacles::GetC1(){ return C1_;}

Matrix_t HomographyObstacles::GetC2(){ return C2_;}

void HomographyObstacles::SetBobs(const FPTYPE c3, const int iteration, const bool isObstacle){
    Matrix_t Bobs(N_, 1);
    Bobs.setZero();

    if(isObstacle){
        for(int i=0; i<N_; i++){
            Bobs(i, 0) = c3;
        }
    }

    Bobs_ = Bobs;

}

Matrix_t HomographyObstacles::GetBobs(){ return Bobs_; }

Matrix_t HomographyObstacles::ObstacleRestrictionsA(const Matrix_t &Ppu){

    Matrix_t C( N_, 2 * N_ + 2 * m_);
    C.setZero();

    C.block(0 ,0, N_, N_) = C1_ * Ppu;
    C.block(0,N_ + m_, N_, N_) = C2_ * Ppu;


    return C; 
}

Matrix_t HomographyObstacles::ObstacleBoundaryVector(const Matrix_t &Pps, const Matrix_t &x_state, const Matrix_t &y_state){

    ////std::cout << " Bobs_ - Eobs_*PpsVector \n" <<  Bobs_ - Eobs_*PpsVector << std::endl;
    //std::cout << " Bobs_ - Eobs_*PpsVector \n" <<   << std::endl;
    return (Bobs_ - C1_ * Pps * x_state - C2_ * Pps * y_state);
}

Matrix_t HomographyObstacles::GetA(){
    Matrix_t Pzu = mpc_model_.get_Pzu();
    Matrix_t Ppu = mpc_model_.get_Ppu();
    Matrix_t U_future = step_generator_.GetFuture_U_Matrix();

    const Matrix_t &trunk_orientation = rotation_controller_->GetTrunkOrientation();
    const Matrix_t &feet_orientation  = rotation_controller_->GetFeetOrientation();

    Matrix_t feet = robot_restrictions_.GetFeetRestrictionsMatrix(feet_orientation);
    Matrix_t feetA(2*feet.rows(),feet.cols()); feetA.setZero();
    feetA.block(0,0,feet.rows(),feet.cols()) = feet;
    feetA.block(feet.rows(),0,feet.rows(),feet.cols()) = -feet;


    // For ZMP restrictions.
    Matrix_t Uc = step_generator_.GetCurrent_U_Matrix();
    Matrix_t Uf = step_generator_.GetFuture_U_Matrix();
    unsigned short int s1 = step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
    unsigned short int s2 = step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));
    Matrix_t SmVector(m_+1,1); SmVector.setZero();
    SmVector(0,0) = s1; SmVector(1,0) = s2; SmVector(2,0) = N_;
    Matrix_t thetas_support(m_+1,1); thetas_support.setZero();
    thetas_support(0,0) = rotation_controller_->GetSupportFootOrientation();
    thetas_support(1,0) = feet_orientation(s1-1,0);
    thetas_support(2,0) = feet_orientation(s2-1,0);
    Matrix_t zmp = robot_restrictions_.GetZMPRestrictionsMatrix(thetas_support, Pzu, U_future,SmVector);
    Matrix_t zmpA(2*zmp.rows(),zmp.cols()); zmpA.setZero();
    zmpA.block(0,0,zmp.rows(),zmp.cols()) = zmp;
    zmpA.block(zmp.rows(),0,zmp.rows(),zmp.cols()) = -zmp;

    Matrix_t obstacle = ObstacleRestrictionsA(Ppu);

    Matrix_t A(feetA.rows() + zmpA.rows() + obstacle.rows(), feet.cols());
    A.setZero();
    A.block(0, 0, feetA.rows(), feetA.cols()) = feetA;
    A.block(feetA.rows(), 0, zmpA.rows(), zmpA.cols()) = zmpA;
    A.block(feetA.rows() + zmpA.rows(),0,obstacle.rows(),obstacle.cols()) = obstacle;

    //std::cout << "A_obstacle: \n" << A  << std::endl; 

    return A;
}

Matrix_t HomographyObstacles::GetubA(){
    const Matrix_t &trunk_orientation = rotation_controller_->GetTrunkOrientation();
    const Matrix_t &feet_orientation  = rotation_controller_->GetFeetOrientation();

    Matrix_t Pzs = mpc_model_.get_Pzs();
    Matrix_t Pps = mpc_model_.get_Pps();
    Matrix_t U_current = step_generator_.GetCurrent_U_Matrix();
    Matrix_t x_state = dynamic_model_.GetXStateVector();
    Matrix_t y_state = dynamic_model_.GetYStateVector();

    auto feet = robot_restrictions_.GetFeetUpperBoundaryVector(feet_orientation(0,0), support_foot_position_, step_generator_.GetSupportPhase(), step_generator_.GetFlyingFoot());
    Matrix_t feetUB(2 * feet.rows(), 1); feetUB.setZero();
    auto feet_lower = robot_restrictions_.GetFeetLowerBoundaryVector(feet_orientation(0,0), support_foot_position_, step_generator_.GetSupportPhase(), step_generator_.GetFlyingFoot());
    feetUB.block(0,0,feet.rows(),1) = feet;
    feetUB.block(feet.rows(),0,feet_lower.rows(),1) = -feet_lower;


    // For ZMP restrictions.
    Matrix_t Uc = step_generator_.GetCurrent_U_Matrix();
    Matrix_t Uf = step_generator_.GetFuture_U_Matrix();
    unsigned short int s1 = step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
    unsigned short int s2 = step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));
    Matrix_t SmVector(m_+1,1); SmVector.setZero();
    SmVector(0,0) = s1; SmVector(1,0) = s2; SmVector(2,0) = N_;
    Matrix_t thetas_support(m_+1,1); thetas_support.setZero();
    thetas_support(0,0) = rotation_controller_->GetSupportFootOrientation();
    thetas_support(1,0) = feet_orientation(s1-1,0);
    thetas_support(2,0) = feet_orientation(s2-1,0);
    auto zmp = robot_restrictions_.GetZMPUpperBoundaryVector(support_foot_position_, thetas_support, Pzs, step_generator_, x_state, y_state,SmVector);
    Matrix_t zmpUB(2 * zmp.rows(),1); zmpUB.setZero();
    auto zmp_lower = robot_restrictions_.GetZMPLowerBoundaryVector(support_foot_position_, thetas_support, Pzs, step_generator_, x_state, y_state,SmVector);  
    zmpUB.block(0,0,zmp.rows(),1) = zmp;  
    zmpUB.block(zmp.rows(),0,zmp_lower.rows(),1) = -zmp_lower;  


    Matrix_t obstacle = ObstacleBoundaryVector(Pps, x_state, y_state);

    Matrix_t ubA(feetUB.rows() + zmpUB.rows() + obstacle.rows(), 1);
    ubA.block(0, 0, feetUB.rows(), feetUB.cols()) = feetUB;
    ubA.block(feetUB.rows(), 0, zmpUB.rows(), zmpUB.cols()) = zmpUB;
    ubA.block(feetUB.rows() + zmpUB.rows(),0,obstacle.rows(),obstacle.cols()) = obstacle;

    //std::cout << "ubA\n" << ubA  << std::endl; 
    //std::cout << "\nobstacle \n" <<  obstacle << std::endl; 

    return ubA;
}

Matrix_t HomographyObstacles::GetlbA(){
    // const Matrix_t &trunk_orientation = rotation_controller_->GetTrunkOrientation();
    // const Matrix_t &feet_orientation  = rotation_controller_->GetFeetOrientation();

    // Matrix_t Pzs = mpc_model_.get_Pzs();
    // Matrix_t U_current = step_generator_.GetCurrent_U_Matrix();
    // Matrix_t x_state = dynamic_model_.GetXStateVector();
    // Matrix_t y_state = dynamic_model_.GetYStateVector();

    // auto feet = robot_restrictions_.GetFeetLowerBoundaryVector(feet_orientation(0,0), support_foot_position_, step_generator_.GetSupportPhase(), step_generator_.GetFlyingFoot());

    // // For ZMP restrictions.
    // Matrix_t Uc = step_generator_.GetCurrent_U_Matrix();
    // Matrix_t Uf = step_generator_.GetFuture_U_Matrix();
    // unsigned short int s1 = step_generator_.GetSelectionIndex(Uf.block(0,0,N_,1));
    // unsigned short int s2 = step_generator_.GetSelectionIndex(Uf.block(0,1,N_,1));
    // Matrix_t SmVector(m_+1,1); SmVector.setZero();
    // SmVector(0,0) = s1; SmVector(1,0) = s2; SmVector(2,0) = N_;
    // Matrix_t thetas_support(m_+1,1); thetas_support.setZero();
    // thetas_support(0,0) = rotation_controller_->GetSupportFootOrientation();
    // thetas_support(1,0) = feet_orientation(s1-1,0);
    // thetas_support(2,0) = feet_orientation(s2-1,0);
    // auto zmp = robot_restrictions_.GetZMPLowerBoundaryVector(support_foot_position_, thetas_support, Pzs, step_generator_, x_state, y_state,SmVector);

    // Matrix_t lbA(feet.rows() + zmp.rows() + 3 * N_, 1);
    // lbA.setZero();
    // lbA.block(0, 0, feet.rows(), feet.cols()) = feet;
    // lbA.block(feet.rows(), 0, zmp.rows(), zmp.cols()) = zmp;
    // lbA.block(feet.rows() + zmp.rows(),0,3 * N_,1) = Matrix_t::Constant( 3 * N_, 1, -15);

    //std::cout << "lbA\n" << lbA  << std::endl; 
    Matrix_t lbA;
    return lbA;
}

HomographyNonLinearSimulated::HomographyNonLinearSimulated(const boost::property_tree::ptree &parameters) : HomographySimulated(parameters),
alpha_R_trunk_(parameters.get<FPTYPE>("qp.alpha_R_trunk")),
alpha_R_foot_(parameters.get<FPTYPE>("qp.alpha_R_foot")),
gammaR_(parameters.get<FPTYPE>("qp.gammaR")),
betaR_(parameters.get<FPTYPE>("qp.betaR"))
{
    // Get the first initial position-reference position homography
    const Matrix_t &homography_matrix = homography_matrices_[first_reference_image_];
    Matrix_t R; Vector3D_t t,n;
    FPTYPE d;
    // Decompose the first homography
    RecoverFromHomography(homography_matrix,R,t,n,d,current_iteration_,homography_solution_);
    // Deduce the angle to follow

    RefAnglesInterpolation(-common::YRotationAngleFromRotationMatrix(R));
}

Matrix_t HomographyNonLinearSimulated::GetH()
{
    Matrix_t Hessian(4*N_ + 2*m_, 4*N_ + 2*m_);
    Hessian.setZero();
    Matrix_t Q_prime(N_+m_, N_+m_);
    Q_prime.setZero();
    const Matrix_t &Pvu = mpc_model_.get_Pvu();
    const Matrix_t &Pzu = mpc_model_.get_Pzu();
    const Matrix_t &Ppu = mpc_model_.get_Ppu();
    const Matrix_t &Pau = mpc_model_.get_Pau();
    const Matrix_t &Pps = mpc_model_.get_Pps();

    const Matrix_t &xcom_state = dynamic_model_.GetXStateVector();
    const Matrix_t &ycom_state = dynamic_model_.GetYStateVector();
    const Matrix_t &tcom_state = dynamic_model_.GetTCOMStateVector();
    const Matrix_t &tfeet_state= dynamic_model_.GetTFOOTStateVector();

    Matrix_t Xk1   = Pps*xcom_state + Ppu*u_0_.block(  0      , 0, N_, 1);
    Matrix_t Yk1   = Pps*ycom_state + Ppu*u_0_.block(  N_+  m_, 0, N_, 1);
    Matrix_t thetas= Pps*tcom_state + Ppu*u_0_.block(3*N_+2*m_, 0, N_, 1);

    // TODO: PRE-COMPUTE THE CONSTANT MATRICES (Low priority)
    Matrix_t U_future = step_generator_.GetFuture_U_Matrix();
    Q_prime.block(0, 0, N_, N_) = alpha_*Matrix_t::Identity(N_, N_) + eta_x_ * Pvu.transpose() * Pvu +
                                  gamma_ * Pzu.transpose() * Pzu + kappa_ * Pau.transpose() * Pau;
    Q_prime.block(0, N_, N_, m_) = -gamma_*Pzu.transpose()*U_future;
    Q_prime.block(N_, 0, m_, N_) = -gamma_*U_future.transpose()*Pzu;
    Q_prime.block(N_, N_, m_, m_) = gamma_*U_future.transpose()*U_future;
    Hessian.block(0, 0, N_+m_, N_+m_) = Q_prime;
    Q_prime.block(0, 0, N_, N_) = alpha_*Matrix_t::Identity(N_, N_) + eta_y_ * Pvu.transpose() * Pvu +
                                  gamma_ * Pzu.transpose() * Pzu + kappa_ * Pau.transpose() * Pau;
    Hessian.block(N_+m_, N_+m_, N_+m_, N_+m_) = Q_prime;
    const Matrix_t &Ppuk = mpc_model_.get_Ppuk();
    Hessian.block(2*N_+2*m_, 2*N_+2*m_, N_, N_) =  alpha_R_foot_*Matrix_t::Identity(N_, N_)+ gammaR_*Ppuk.transpose() * Ppuk; //Foot
    Hessian.block(3*N_+2*m_, 3*N_+2*m_, N_, N_) =  alpha_R_trunk_*Matrix_t::Identity(N_, N_)+  betaR_*Ppu.transpose() * Ppu;//Trunk
    for (int i = first_reference_image_; i < last_reference_image_ && i - first_reference_image_ < REFERENCE_WINDOW_; i++) {
        std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];
        FPTYPE error_gain = error_gains_[i - first_reference_image_];

        for (VisualFeatureData &visual_feature : current_visual_features) {
            MPCNonLinearHomography *hom_nlmpc = dynamic_cast<MPCNonLinearHomography *>(visual_feature.model.get());
            if (!hom_nlmpc)
                throw std::runtime_error("ERROR: the pointer is not MPCNonLinearEssential*");
            Matrix_t HX = hom_nlmpc->HX(thetas);        // SIN  in Noe's code
            Matrix_t HY = hom_nlmpc->HY(thetas);        // COS  in Noe's code
            Matrix_t HT = hom_nlmpc->HT(Xk1,Yk1,thetas);// GRADCOS/GRASIN in Noe's code

            const FPTYPE &beta = visual_feature.model->GetGain();
            Hessian.block(0, 0, N_, N_)           += error_gain * beta*Ppu.transpose() * HX * HX * Ppu;
            Hessian.block(0, N_ + m_, N_, N_)     += error_gain * beta*Ppu.transpose() * HX * HY * Ppu;
            Hessian.block(0, 3*N_ + 2*m_, N_, N_) += error_gain * beta*Ppu.transpose() * HX * HT * Ppu;

            Hessian.block(N_+m_,0, N_, N_)        += error_gain * beta*Ppu.transpose() * HY * HX * Ppu;
            Hessian.block(N_+m_,N_+m_, N_, N_)    += error_gain * beta*Ppu.transpose() * HY * HY * Ppu;
            Hessian.block(N_+m_,3*N_+2*m_,N_,N_)  += error_gain * beta*Ppu.transpose() * HY * HT * Ppu;

            Hessian.block(3*N_+2*m_,0, N_, N_)          += error_gain * beta*Ppu.transpose() * HT * HX * Ppu;
            Hessian.block(3*N_+2*m_,N_ + m_, N_, N_)    += error_gain * beta*Ppu.transpose() * HT * HY * Ppu;
            Hessian.block(3*N_+2*m_,3*N_ + 2*m_, N_, N_)+= error_gain * beta*Ppu.transpose() * HT * HT * Ppu;
        }
    }
    return Hessian;
}

Matrix_t HomographyNonLinearSimulated::Getg()
{
    Matrix_t g(4*N_ + 2*m_, 1);
    g.setZero();

    const Matrix_t &Pvu = mpc_model_.get_Pvu();
    const Matrix_t &Pzu = mpc_model_.get_Pzu();
    const Matrix_t &Ppu = mpc_model_.get_Ppu();
    const Matrix_t &Pzs = mpc_model_.get_Pzs();
    const Matrix_t &Pvs = mpc_model_.get_Pvs();
    const Matrix_t &Pps = mpc_model_.get_Pps();
    const Matrix_t &Pau = mpc_model_.get_Pau();
    const Matrix_t &Pas = mpc_model_.get_Pas();

    Matrix_t U_future = step_generator_.GetFuture_U_Matrix();
    Matrix_t U_current = step_generator_.GetCurrent_U_Matrix();

    const Matrix_t &xcom_state = dynamic_model_.GetXStateVector();
    const Matrix_t &ycom_state = dynamic_model_.GetYStateVector();
    const Matrix_t &tcom_state = dynamic_model_.GetTCOMStateVector();
    const Matrix_t &tfeet_state= dynamic_model_.GetTFOOTStateVector();

    Matrix_t Xk1   = Pps*xcom_state + Ppu*u_0_.block(  0      , 0, N_, 1);
    Matrix_t Yk1   = Pps*ycom_state + Ppu*u_0_.block(  N_+  m_, 0, N_, 1);
    Matrix_t thetas= Pps*tcom_state + Ppu*u_0_.block(3*N_+2*m_, 0, N_, 1);

    // TODO: CHECK AND COMPUTE THE CONSTANT MATRICES SEPARATELY (low priority)
    g.block(0, 0, N_, 1) = eta_x_ * Pvu.transpose() * Pvs * xcom_state +
                           gamma_*Pzu.transpose() * (Pzs*xcom_state - U_current * X(support_foot_position_)) +
                           kappa_ * Pau.transpose() * Pas * xcom_state ;
    g.block(N_, 0, m_, 1) = -gamma_*U_future.transpose()*(Pzs*xcom_state - U_current * X(support_foot_position_));

    g.block(N_+m_, 0, N_, 1) = eta_y_ * Pvu.transpose() * Pvs * ycom_state +
                               gamma_ * Pzu.transpose() * (Pzs*ycom_state- U_current*Y(support_foot_position_)) +
                               kappa_ * Pau.transpose() * Pas * ycom_state;

    g.block(N_+m_+N_, 0, m_, 1) = -gamma_*U_future.transpose()*(Pzs*ycom_state - U_current*Y(support_foot_position_));

    const Matrix_t &Ppuk = mpc_model_.get_Ppuk();
    const Matrix_t &thetaV = mpc_model_.get_ThetaV();
    g.block(2*N_+2*m_, 0, N_, 1) = gammaR_*Ppuk.transpose()*(thetaV - thetaRef_); // (thetaV - theta_ref_foot_);
    g.block(3*N_+2*m_, 0, N_, 1) = betaR_*Ppu.transpose()*(Pps*tcom_state - thetaRef_);

    for (int i = first_reference_image_; i < last_reference_image_ && i - first_reference_image_ < REFERENCE_WINDOW_; i++) {
        std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];
        const FPTYPE &error_gain = error_gains_[i - first_reference_image_];
        for (VisualFeatureData &visual_feature : current_visual_features) {
            const FPTYPE &actual    = visual_feature.actual;
            const FPTYPE &predicted = visual_feature.predicted;
            FPTYPE desired_value = visual_feature.model->GetExpectedValue() - (actual - predicted);
            MPCNonLinearHomography *hom_nlmpc = dynamic_cast<MPCNonLinearHomography *>(visual_feature.model.get());
            if (!hom_nlmpc)
                throw std::runtime_error("ERROR: the pointer is not MPCNonLinearHomography*");
            Matrix_t RE = visual_feature.model->GetGain()*(hom_nlmpc->RE(Xk1,Yk1,thetas)-Matrix_t::Constant(N_, 1, desired_value));
            Matrix_t gcontrib(4*N_ + 2*m_,1); gcontrib.setZero();
            const FPTYPE &beta = visual_feature.model->GetGain();
            gcontrib.block(0, 0, N_, 1)       += error_gain*beta*Ppu.transpose()*hom_nlmpc->HX(thetas)*RE;
            gcontrib.block(N_+m_,0,N_, 1)     += error_gain*beta*Ppu.transpose()*hom_nlmpc->HY(thetas)*RE;
            gcontrib.block(3*N_+2*m_,0,N_, 1) += error_gain*beta*Ppu.transpose()*hom_nlmpc->HT(Xk1,Yk1,thetas)*RE;
            g += gcontrib;
        }
    }

    Matrix_t Hessian(4*N_ + 2*m_, 4*N_ + 2*m_);
    Hessian.setZero();
    Matrix_t Q_prime(N_+m_, N_+m_);
    Q_prime.setZero();
    Q_prime.block(0, 0, N_, N_) = alpha_*Matrix_t::Identity(N_, N_) + eta_x_ * Pvu.transpose() * Pvu +
                                  gamma_ * Pzu.transpose() * Pzu + kappa_ * Pau.transpose() * Pau;
    Q_prime.block(0, N_, N_, m_) = -gamma_*Pzu.transpose()*U_future;
    Q_prime.block(N_, 0, m_, N_) = -gamma_*U_future.transpose()*Pzu;
    Q_prime.block(N_, N_, m_, m_) = gamma_*U_future.transpose()*U_future;
    Hessian.block(0, 0, N_+m_, N_+m_) = Q_prime;
    Q_prime.block(0, 0, N_, N_) = alpha_*Matrix_t::Identity(N_, N_) + eta_y_ * Pvu.transpose() * Pvu +
                                  gamma_ * Pzu.transpose() * Pzu + kappa_ * Pau.transpose() * Pau;
    Hessian.block(N_+m_, N_+m_, N_+m_, N_+m_) = Q_prime;

    Hessian.block(2*N_+2*m_, 2*N_+2*m_, N_, N_) =  alpha_R_foot_*Matrix_t::Identity(N_, N_)+ gammaR_*Ppuk.transpose() * Ppuk; //Foot
    Hessian.block(3*N_+2*m_, 3*N_+2*m_, N_, N_) =  alpha_R_trunk_*Matrix_t::Identity(N_, N_)+  betaR_*Ppu.transpose() * Ppu; //Trunk
    g += Hessian.transpose() * u_0_;
    return g;
}


Matrix_t HomographyNonLinearSimulated::Getlb()
{
    // No constraints, return empty vector
    Matrix_t lb;
    return lb;
}

Matrix_t HomographyNonLinearSimulated::Getub()
{
    // No constraints, return empty vector
    Matrix_t ub;
    return ub;
}