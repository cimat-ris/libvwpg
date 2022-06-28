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

// Constructor
EssentialBase::EssentialBase(const boost::property_tree::ptree &parameters):
    VisualFeatureBase(parameters), h(parameters.get<FPTYPE>("qp.estimated_virtual_height")),
    e11_predicted_(1.0),e13_predicted_(1.0),e22_predicted_(0.0),e31_predicted_(-1.0),e33_predicted_(1.0)
{ 
    // Initialize the names of the visual features to be used (h13, h33)
    common::logString(": initialize visual features ");
    all_visual_features_names.push_back("e32");
    all_visual_features_names.push_back("e12");
    all_visual_features_names.push_back("e21");
    all_visual_features_names.push_back("e23");
    for (auto visual_feature_name : all_visual_features_names) {
        latest_visual_data_[visual_feature_name] = 0.0;
        expected_values[visual_feature_name]     = 0.0;
        std::ostringstream oss;
        oss << ": Adding feature " << visual_feature_name;
        common::logString(oss.str());
    }
}

// Rotation reference is set with respect to the first target image
void EssentialBase::SolveOrientation()
{
    // Get the first initial position-virtual reference position essential matrix
    const Matrix_t &essential_matrix = essential_matrices_[first_reference_image_];
    Matrix_t R; Vector3D_t t;
    // Decompose the essential matrix to the first target image
    RecoverFromEssential(essential_matrix,R,t);
    // Deduce the angle to target
    FPTYPE target_angle = -common::YRotationAngleFromRotationMatrix(R);
    // Pass it to the rotation controller as a reference angle
    rotation_controller_->UpdateReference(target_angle);
    // Performs the optimization for the next step
    rotation_controller_->ComputeRobotNextOrientation();
}

// Compute the series of essential matrices from the current points to the reference points
void EssentialBase::SimulateVisualFeatures()
{
    // Cycle over the subsequent target images
    for (int i = first_reference_image_; i < last_reference_image_; i++) {
        // Get the corresponding target image points
        InterestPoints &current_interest_points = all_virtual_interest_points_[i];
        InterestPoints &current_interest_points_gt = all_ground_truth_virtual_interest_points_[i];
        std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];

        // Ground truth section.
        const std::vector<Point2D_t> &reference_image_points_gt = current_interest_points_gt.GetInReferenceImageCoordinates();
        const std::vector<Point2D_t> &current_image_points_gt   = current_interest_points_gt.SimulatedProjection(camera_ground_truth_);
        const std::vector<bool>        &current_visibility_gt   = current_interest_points_gt.GetVisibility();

        std::vector<cv::Point2d> cv_reference_points_gt;
        std::vector<cv::Point2d> cv_current_points_gt;
        OpencvVector(reference_image_points_gt,current_image_points_gt,current_visibility_gt,cv_reference_points_gt,cv_current_points_gt);

        Matrix_t essential_matrix_gt = ComputeEssential(cv_reference_points_gt, cv_current_points_gt,camera_ground_truth_.GetIntrisicParametersMatrix());
        essential_matrix_gt =  (1.0 / essential_matrix_gt(0, 2)) * essential_matrix_gt;
        essential_matrices_ground_truth_[i] = essential_matrix_gt;

        //Simulated section
        // Reference points in image coordinates
        const std::vector<Point2D_t> &reference_image_points = current_interest_points.GetInReferenceImageCoordinates();
        const std::vector<Point2D_t> &current_image_points   = current_interest_points.SimulatedProjection(camera_ground_truth_,true);
        const std::vector<bool>        &current_visibility   = current_interest_points.GetVisibility();

        if (i == first_reference_image_) {
            const std::vector<Point3D_t> &reference_world_points = current_interest_points.GetInCurrentWorldCoordinates();
            for (int k=0;k<current_image_points.size();k++) {
                common::Log3DInformation("[current_world_point]", current_iteration_, i, k,  reference_world_points[k]);
                common::Log2DInformation("[current_image_point]", current_iteration_, i, k, current_image_points[k], current_visibility[k]);
            }
        }

        std::vector<cv::Point2d> cv_reference_points;
        std::vector<cv::Point2d> cv_current_points;
        OpencvVector(reference_image_points,current_image_points,current_visibility,cv_reference_points,cv_current_points);

        Matrix_t essential_matrix = Matrix_t::Identity(3,3);
        if (cv_reference_points.size()>min_points_to_estimate_constraint_) {// Computes the homography from current to reference image points if there are enough points
            // Computes the essential matrix from current to reference image points
            essential_matrix = ComputeEssential(cv_reference_points, cv_current_points,camera_.GetIntrisicParametersMatrix());
            essential_matrix = (1.0 / essential_matrix(0, 2)) * essential_matrix;
        }
        else{
            // When not enough features are available, use the current prediction
            for (VisualFeatureData &visual_feature : current_visual_features) {
                // A visual feature is associated to position in the homography matrix
                std::pair<int,int> p = visual_feature.model->GetMatrixPosition();
                // Actual and predicted values of the visual feature
                essential_matrix(p.first, p.second) = visual_feature.predicted;
            }
            essential_matrix(0,0) = e11_predicted_;
            essential_matrix(0,2) = e13_predicted_;
            essential_matrix(1,1) = e22_predicted_;
            essential_matrix(2,0) = e31_predicted_;
            essential_matrix(2,2) = e33_predicted_;
        }
        essential_matrices_[i] = essential_matrix;
  }
}

// Compute the Hessian matrix H (for the QP solver in the linear case)
Matrix_t EssentialBase::GetH()
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
    Q_prime.block(0, 0, N_, N_)   = alpha_*Matrix_t::Identity(N_, N_) + eta_x_ * Pvu.transpose() * Pvu +
                                    gamma_ * Pzu.transpose() * Pzu + kappa_ * Pau.transpose() * Pau;
    Q_prime.block(0, N_, N_, m_)  = -gamma_*Pzu.transpose()*U_future;
    Q_prime.block(N_, 0, m_, N_)  = -gamma_*U_future.transpose()*Pzu;
    Q_prime.block(N_, N_, m_, m_) = gamma_*U_future.transpose()*U_future;  
    Q_hat.setZero();


    switch(multiple_objective_method_){
        case MultipleObjectives::WeightedAverages :
            // Cycle over the reference images
            for (int i = first_reference_image_; i < last_reference_image_ && i - first_reference_image_ < REFERENCE_WINDOW_; i++) {
                std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];
                FPTYPE error_gain = error_gains_[i - first_reference_image_];

                for (VisualFeatureData &visual_feature : current_visual_features) {
                    MPCLinearEssential *ess_mpc = dynamic_cast<MPCLinearEssential *>(visual_feature.model.get());
                    if (!ess_mpc)
                      throw std::runtime_error("ERROR: the pointer is not MPCLinearEssential*");
                    Matrix_t A = ess_mpc->A();
                    Matrix_t B = ess_mpc->B();
                    const FPTYPE &beta = visual_feature.model->GetGain();
                    Q_hat.block(0, 0, N_, N_)            += error_gain * beta*Ppu.transpose() * A.transpose() * A * Ppu;
                    Q_hat.block(0, N_ + m_, N_, N_)      += error_gain * beta*Ppu.transpose() * A.transpose() * B * Ppu;
                    Q_hat.block(N_ + m_, 0 , N_, N_)     += error_gain * beta*Ppu.transpose() * B.transpose() * A * Ppu;
                    Q_hat.block(N_ + m_, N_ + m_, N_, N_)+= error_gain * beta*Ppu.transpose() * B.transpose() * B * Ppu;
                }
            }
        case MultipleObjectives::SharedPredictionWindows :
            // Cycle over the reference images
            for (int i = first_reference_image_; i < last_reference_image_ && i - first_reference_image_ < REFERENCE_WINDOW_; i++) {
                std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];
                std::vector<VisualFeatureData> &current_visual_features_Ds = all_visual_features_[first_reference_image_];
                Matrix_t W;
                if ((i == 0) && (current_iteration_ == 0)) {
                    W = Matrix_t::Identity(N_, N_);
                } else if ((i == 1) && (current_iteration_ == 0))
                    W = Matrix_t::Zero(N_, N_);
                else if (i == first_reference_image_ && current_iteration_ != 0 &&
                         (first_reference_image_ + 1 - last_reference_image_ != 0)) {
                    Matrix_t Ds = GetDs(current_visual_features_Ds);
                    W = Ds.transpose() * Ds;
                } else if (i == first_reference_image_ + 1 && current_iteration_ != 0 &&
                           (first_reference_image_ + 1 - last_reference_image_ != 0)) {
                    Matrix_t I = Matrix_t::Identity(N_, N_);
                    Matrix_t Ds = GetDs(current_visual_features_Ds);
                    Matrix_t IminDs = I - Ds;
                    W = IminDs.transpose() * IminDs;
                } else
                    W = Matrix_t::Identity(N_, N_);

                for (VisualFeatureData &visual_feature : current_visual_features) {
                    MPCLinearEssential *ess_mpc = dynamic_cast<MPCLinearEssential *>(visual_feature.model.get());
                    if (!ess_mpc)
                        throw std::runtime_error("ERROR: the pointer is not MPCLinearEssential*");
                    Matrix_t A = ess_mpc->A();
                    Matrix_t B = ess_mpc->B();
                    const FPTYPE &beta = visual_feature.model->GetGain();
                    Q_hat.block(0, 0, N_, N_)            +=  beta*Ppu.transpose() * A.transpose() * W * A * Ppu;
                    Q_hat.block(0, N_ + m_, N_, N_)      +=  beta*Ppu.transpose() * A.transpose() * W * B * Ppu;
                    Q_hat.block(N_ + m_, 0 , N_, N_)     +=  beta*Ppu.transpose() * B.transpose() * W * A * Ppu;
                    Q_hat.block(N_ + m_, N_ + m_, N_, N_)+=  beta*Ppu.transpose() * B.transpose() * W * B * Ppu;
                }

            }
            break;
        default:
            break;
    }

    H.setZero();
    H.block(0, 0, N_+m_, N_+m_) = Q_prime;
    Q_prime.block(0, 0, N_, N_)   = alpha_*Matrix_t::Identity(N_, N_) + eta_y_ * Pvu.transpose() * Pvu +
                                    gamma_ * Pzu.transpose() * Pzu + kappa_ * Pau.transpose() * Pau;
    H.block(N_+m_, N_+m_, N_+m_, N_+m_) = Q_prime;
    return H + Q_hat;
}

// Compute the vector g (for the QP solver in the linear case)
Matrix_t EssentialBase::Getg()
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

    g.block(N_+m_, 0, N_, 1) = eta_y_ * Pvu.transpose() * (Pvs * y_state )+
                               gamma_ * Pzu.transpose() * (Pzs*y_state - U_current*Y(support_foot_position_)) +
                               kappa_ * Pau.transpose() * Pas * y_state;
    g.block(N_+m_+N_, 0, m_, 1) = -gamma_*U_future.transpose()*(Pzs*y_state - U_current*Y(support_foot_position_));
    
    p_hat.setZero();

    switch(multiple_objective_method_){
        case MultipleObjectives::WeightedAverages :
            for (int i = first_reference_image_; i < last_reference_image_ && i - first_reference_image_ < REFERENCE_WINDOW_; i++) {
                std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];
                const FPTYPE &error_gain = error_gains_[i - first_reference_image_];

                for (VisualFeatureData &visual_feature : current_visual_features) {
                    MPCLinearEssential *ess_mpc = dynamic_cast<MPCLinearEssential *>(visual_feature.model.get());
                    if (!ess_mpc)
                      throw std::runtime_error("ERROR: the pointer is not MPCLinearEssential*");
                    Matrix_t A = ess_mpc->A();
                    Matrix_t B = ess_mpc->B();
                    Matrix_t C = ess_mpc->C();
                    const FPTYPE &beta      = visual_feature.model->GetGain();
                    const FPTYPE &actual    = visual_feature.actual;
                    const FPTYPE &predicted = visual_feature.predicted;
                    FPTYPE desired_value = visual_feature.model->GetExpectedValue() - (actual - predicted);

                    Matrix_t common = A * Pps * x_state + B * Pps * y_state + C - Matrix_t::Constant(N_, 1, desired_value);
                    p_hat.block(0, 0, N_, 1)       += error_gain * beta*Ppu.transpose() * A.transpose() * common;
                    p_hat.block(N_ + m_, 0, N_, 1) += error_gain * beta*Ppu.transpose() * B.transpose() * common;
                }
            }
            break;
        case MultipleObjectives::SharedPredictionWindows:
            // Cycle over the reference images
            for (int i = first_reference_image_; i < last_reference_image_ && i - first_reference_image_ < REFERENCE_WINDOW_; i++) {
                std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];
                std::vector<VisualFeatureData> &current_visual_features_Ds = all_visual_features_[first_reference_image_];
                Matrix_t W;
                if (((i == 0) && (current_iteration_ == 0))) {
                    W = Matrix_t::Identity(N_, N_);
                } else if ((i == 1) && (current_iteration_ == 0))
                    W = Matrix_t::Zero(N_, N_);
                else if (i == first_reference_image_ && current_iteration_ != 0 &&
                         (first_reference_image_ + 1 - last_reference_image_ != 0)) {
                    Matrix_t Ds = GetDs(current_visual_features_Ds);
                    W = Ds.transpose() * Ds;
                } else if (i == first_reference_image_ + 1 && current_iteration_ != 0 &&
                           (first_reference_image_ + 1 - last_reference_image_ != 0)) {
                    Matrix_t I = Matrix_t::Identity(N_, N_);
                    Matrix_t Ds = GetDs(current_visual_features_Ds);
                    Matrix_t IminDs = I - Ds;
                    W = IminDs.transpose() * IminDs;
                } else
                    W = Matrix_t::Identity(N_, N_);

                for (VisualFeatureData &visual_feature : current_visual_features) {
                    MPCLinearEssential *ess_mpc = dynamic_cast<MPCLinearEssential *>(visual_feature.model.get());
                    if (!ess_mpc)
                        throw std::runtime_error("ERROR: the pointer is not MPCLinearEssential*");
                    Matrix_t A = ess_mpc->A();
                    Matrix_t B = ess_mpc->B();
                    Matrix_t C = ess_mpc->C();
                    const FPTYPE &beta      = visual_feature.model->GetGain();
                    const FPTYPE &actual    = visual_feature.actual;
                    const FPTYPE &predicted = visual_feature.predicted;
                    FPTYPE desired_value = visual_feature.model->GetExpectedValue() - (actual - predicted);

                    Matrix_t common = A * Pps * x_state + B * Pps * y_state + C - Matrix_t::Constant(N_, 1, desired_value);
                    p_hat.block(0, 0, N_, 1)       +=   beta*Ppu.transpose() * A.transpose() * W * common;
                    p_hat.block(N_ + m_, 0, N_, 1) +=   beta*Ppu.transpose() * B.transpose() * W * common;
                }

            }
            break;
        default:
            break;
    }

    return g + p_hat;
}

Matrix_t EssentialBase::GetDs(std::vector<VisualFeatureData> &current_visual_features) {
    std::vector<int > indices;
    indices.clear();
    int index;
    int value_index = 0;
    int counter = 0;
    for (VisualFeatureData &visual_feature : current_visual_features) {
        MPCLinearEssential *ess_mpc = dynamic_cast<MPCLinearEssential *>(visual_feature.model.get());
        if (!ess_mpc)
            throw std::runtime_error("ERROR: the pointer is not MPCLinearEssential*");
        Matrix_t Hp = ess_mpc->GetHp();
        index = GetIndexl(visual_feature.model->GetExpectedValue(),Hp);
        indices.push_back(index);
        if(index<N_)
            counter++;
        if(index<6)
            value_index++;
    }

    int min_index_;
    if(counter>3){
        auto min_index = std::min_element(indices.begin(),indices.end());
        min_index_ = *min_index;
    }
    else
        min_index_ = N_;

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

// Update the visual features from the current frame info
// This code supposes that:
// - the essential matrices have been updated before.
// - the rotation controller has been run, in the linear case.
// - the predicted values from the last step have been calculated before.
void EssentialBase::UpdateActualVisualData()
{
    // For all the target images
    for (int i = first_reference_image_; i < last_reference_image_; i++) {
        // Get the essential matrix and decompose it. 
        const Matrix_t &essential_matrix = essential_matrices_[i];
        const Matrix_t &essential_matrix_gt = essential_matrices_ground_truth_[i];
        Matrix_t R;
        Vector3D_t t;
        RecoverFromEssential(essential_matrix,R,t);
        FPTYPE phi = common::YRotationAngleFromRotationMatrix(R);
        std::ostringstream os;
        os << "[visual_measures]: "
            << "iteration=" << current_iteration_
            << ", phi=" << phi
            << ", t=" << t.transpose();
        common::logString(os.str());
        // Get the visual feature data
        std::vector<VisualFeatureData> &visual_data_for_current_image = all_visual_features_[i];
        for (VisualFeatureData& visual_feature : visual_data_for_current_image) {
            // Update the ck_x_t, ck_z_t parameters, which are the target position in the current camera frame 
            // with z pointing onwards, x on the right. They are easily deduced from the essential matrix.
            visual_feature.model->ck_x_t(X(t));
            visual_feature.model->ck_z_t(Z(t));
            // Update the phi parameter for the next prediction
            visual_feature.model->phi(phi);
            visual_feature.model->c_x_com(c_x_com_);
            visual_feature.model->c_z_com(c_z_com_);
            MPCLinearEssential *ess_mpc = dynamic_cast<MPCLinearEssential *>(visual_feature.model.get());
            if (ess_mpc) {
                // Update the parameters for the next prediction
                ess_mpc->thetas(rotation_controller_->GetTrunkOrientation());
                ess_mpc->ck_y_t(Y(t));
                ess_mpc->y_real(-h);
            } 
            MPCNonLinearEssential *ess_nlmpc = dynamic_cast<MPCNonLinearEssential *>(visual_feature.model.get());
            if (ess_nlmpc) {
                ess_nlmpc->ck_y_t(Y(t));
                ess_nlmpc->y_real(-h);
            } 
            // A visual feature is associated to a position in the essential matrix
            const std::pair<int,int> &p = visual_feature.model->GetMatrixPosition();
            // Get the actual value of the visual feature
            visual_feature.actual    = essential_matrix(p.first, p.second);
            if (i == first_reference_image_) {
                    std::ostringstream oss;
                    oss << "[visual_feature]: "
                      << "iteration=" << current_iteration_
                      << ", name=" << visual_feature.model->GetName()
                      << ", actual=" << visual_feature.actual
                      << ", predicted=" << visual_feature.predicted
                      << ", expected=" << visual_feature.model->GetExpectedValue()
                      << ", ground_truth=" << essential_matrix_gt(p.first, p.second);
                    common::logString(oss.str());
                    // Only keep track of visual data for the first target
                    FPTYPE& current_visual_average = latest_visual_data_[visual_feature.model->GetName()];
                    if (number_of_samples_ > 20)
                        current_visual_average -= current_visual_average / 20;
                    current_visual_average += visual_feature.actual / 20;
            }
        }

        if (i == first_reference_image_) {
            // Some stuff that is done only for the first image
            RefAnglesInterpolation(-phi);
            ++number_of_samples_;
        }
    }
}

// Evaluate the predicted values for the visual features.
void EssentialBase::UpdatePredictedValues(const Matrix_t &solution) {
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
        Xk1 = Pps*xcom_state + Ppu*solution.block(    0   , 0, N_, 1);
        Yk1 = Pps*ycom_state + Ppu*solution.block(  N_+ m_, 0, N_, 1);
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
            MPCLinearEssential *ess_mpc = dynamic_cast<MPCLinearEssential *>(visual_feature.model.get());
            if (ess_mpc){
                Matrix_t A  = ess_mpc->A();
                Matrix_t B  = ess_mpc->B();
                Matrix_t C  = ess_mpc->C();
                Matrix_t Hp = A*Xk1 + B*Yk1 + C;
                ess_mpc->SetHp(Hp);
            }
        }
    }
    const Matrix_t &essential_matrix = essential_matrices_[first_reference_image_];
    Matrix_t R;
    Vector3D_t t;
    RecoverFromEssential(essential_matrix,R,t);
    FPTYPE phi = common::YRotationAngleFromRotationMatrix(R);
    e11_predicted_ = -std::tan(phi+tt);
    e33_predicted_ = -std::tan(phi+tt);
    if (logPredictions_)
        LogCurrentPredictions(solution);
}

// Constructor: Simulated essential matrix (use 3D points)
EssentialSimulated::EssentialSimulated(const boost::property_tree::ptree &parameters):
    EssentialBase(parameters) {

    // World-to-com transformation
    auto com_T_world = world_T_com_.inverse();

    // Initialize reference data. This cycles over target images
    while (true) {
        std::vector<Point3D_t> reference_points_in_world_coordinates;
        // For each target image, load 3d reference points in world coordinates from the ini file
        try {
            reference_points_in_world_coordinates = Load3DReferencePoints_(parameters, com_T_world, last_reference_image_);
        } catch (boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ptree_bad_path> >) {
            break;
        } 
        // Generate the reference points in image coordinates (all_virtual_interest_points_ is formed/completed there)
        InitReferenceImage_(parameters, reference_points_in_world_coordinates, last_reference_image_, true, camera_ground_truth_,false);
        // Generate the reference points in image coordinates for the ground truth (all_ground_truth_interest_points_ is formed/completed there)
        InitReferenceImage_(parameters, reference_points_in_world_coordinates, last_reference_image_, true, camera_ground_truth_,true);
        ++last_reference_image_;
    }
    
    // This is the number of target images
    std::string log_entry(": found " + boost::lexical_cast<std::string>(last_reference_image_) + " reference image in configuration file");
    common::logString(log_entry);

    // Initializes as many essential matrices as target images
    essential_matrices_ = std::vector<Matrix_t>(last_reference_image_);
    essential_matrices_ground_truth_ = std::vector<Matrix_t>(last_reference_image_);

    // Initialize (essential) visual features objects with their model parameters
    // The all_virtual_interest_points_ have been initialized by InitReferenceImage_ above
    common::logString(": initialize interest points ");
    for (InterestPoints points : all_virtual_interest_points_) { 
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
             common::logString(": using a linear formulation of essential matrix entries ");
            // For all visual features (e12,e21...)
            for (auto visual_feature_name : all_visual_features_names ) {
                visual_model_for_reference.push_back(VisualFeatureData(visual_feature_name, N_));
                std::string gain_name = std::string("qp.beta") + visual_feature_name;
                visual_model_for_reference.back().model->gain(parameters.get<FPTYPE>(gain_name));
            }
        } else {
            common::logString(": using a non-linear formulation of essential matrix entries ");
            // For all visual features (e12,e21...)
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

    // Compute the series of essential matrices from the current points to the reference points
    common::logString(": compute essential matrices ");
    SimulateVisualFeatures();

    if (parameters.get<std::string>("qp.linear")=="true"){
        // Solve the first rotation sequence. The reference orientation is set with respect to the first target image
        common::logString(": update reference orientation ");
        SolveOrientation();
    }

    // Update the actual visual features from current frame to allow the first optimization
    common::logString(": update visual data ");
    UpdateActualVisualData();

    // Log the image positions of the reference points in the current position
    TakePictureReferenceFromCurrentPosition();
}

// Constructor: Simulated EssentialNonLinearSimulated (use 3D points)
EssentialNonLinearSimulated::EssentialNonLinearSimulated(const boost::property_tree::ptree &parameters): EssentialSimulated(parameters), 
    alphaR_(parameters.get<FPTYPE>("qp.alphaR")),
    gammaR_(parameters.get<FPTYPE>("qp.gammaR")),
    betaR_(parameters.get<FPTYPE>("qp.betaR"))
{
    const Matrix_t &essential_matrix = essential_matrices_[first_reference_image_];
    Matrix_t R; Vector3D_t t;
    // Decompose the first essential matrix
    RecoverFromEssential(essential_matrix,R,t);
    // Deduce the reference angles to follow
    RefAnglesInterpolation(-common::YRotationAngleFromRotationMatrix(R));
}

// Compute the Hessian matrix
Matrix_t EssentialNonLinearSimulated::GetH() {
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
    Q_prime.block(0, N_, N_, m_)  = -gamma_*Pzu.transpose()*U_future;
    Q_prime.block(N_, 0, m_, N_)  = -gamma_*U_future.transpose()*Pzu;
    Q_prime.block(N_, N_, m_, m_) = gamma_*U_future.transpose()*U_future; 
    Hessian.block(0, 0, N_+m_, N_+m_)         = Q_prime;
    Q_prime.block(0, 0, N_, N_) = alpha_*Matrix_t::Identity(N_, N_) + eta_y_ * Pvu.transpose() * Pvu +
                                  gamma_ * Pzu.transpose() * Pzu + kappa_ * Pau.transpose() * Pau;
    Hessian.block(N_+m_, N_+m_, N_+m_, N_+m_) = Q_prime;
    const Matrix_t &Ppuk = mpc_model_.get_Ppuk();
    Hessian.block(2*N_+2*m_, 2*N_+2*m_, N_, N_) =  alphaR_*Matrix_t::Identity(N_, N_)+ gammaR_*Ppuk.transpose() * Ppuk; //Foot
    Hessian.block(3*N_+2*m_, 3*N_+2*m_, N_, N_) =  alphaR_*Matrix_t::Identity(N_, N_)+  betaR_*Ppu.transpose() * Ppu; //Trunk
    for (int i = first_reference_image_; i < last_reference_image_ && i - first_reference_image_ < REFERENCE_WINDOW_; i++) {
        std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];
        FPTYPE error_gain = error_gains_[i - first_reference_image_];

        for (VisualFeatureData &visual_feature : current_visual_features) {
            MPCNonLinearEssential *ess_nlmpc = dynamic_cast<MPCNonLinearEssential *>(visual_feature.model.get());
            if (!ess_nlmpc)
              throw std::runtime_error("ERROR: the pointer is not MPCNonLinearEssential*");
            Matrix_t HX = ess_nlmpc->HX(thetas);        // diagC  in Noe's code
            Matrix_t HY = ess_nlmpc->HY(thetas);        // diagS  in Noe's code
            Matrix_t HT = ess_nlmpc->HT(Xk1,Yk1,thetas);// diagij in Noe's code

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

Matrix_t EssentialNonLinearSimulated::Getg() {

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
    g.block(2*N_+2*m_, 0, N_, 1) = gammaR_*Ppuk.transpose()*(thetaV - theta_ref_foot_);
    g.block(3*N_+2*m_, 0, N_, 1) = betaR_*Ppu.transpose()*(Pps*tcom_state - thetaRef_); // trunk
    for (int i = first_reference_image_; i < last_reference_image_ && i - first_reference_image_ < REFERENCE_WINDOW_; i++) {
        std::vector<VisualFeatureData> &current_visual_features = all_visual_features_[i];
        const FPTYPE &error_gain = error_gains_[i - first_reference_image_];          
        for (VisualFeatureData &visual_feature : current_visual_features) {
            const FPTYPE &actual    = std::fabs(visual_feature.actual);
            const FPTYPE &predicted = std::fabs(visual_feature.predicted);
            FPTYPE desired_value = visual_feature.model->GetExpectedValue() - (actual - predicted);
            MPCNonLinearEssential *ess_nlmpc = dynamic_cast<MPCNonLinearEssential *>(visual_feature.model.get());
            if (!ess_nlmpc)
              throw std::runtime_error("ERROR: the pointer is not MPCNonLinearEssential*");
            Matrix_t RE = visual_feature.model->GetGain()*(ess_nlmpc->RE(Xk1,Yk1,thetas)-Matrix_t::Constant(N_, 1, desired_value));
            Matrix_t gcontrib(4*N_ + 2*m_,1); gcontrib.setZero();
            const FPTYPE &beta = visual_feature.model->GetGain();
            gcontrib.block(0, 0, N_, 1)       += error_gain*beta*Ppu.transpose()*ess_nlmpc->HX(thetas)*RE;
            gcontrib.block(N_+m_,0,N_, 1)     += error_gain*beta*Ppu.transpose()*ess_nlmpc->HY(thetas)*RE;
            gcontrib.block(3*N_+2*m_,0,N_, 1) += error_gain*beta*Ppu.transpose()*ess_nlmpc->HT(Xk1,Yk1,thetas)*RE;
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

    Hessian.block(2*N_+2*m_, 2*N_+2*m_, N_, N_) =  alphaR_*Matrix_t::Identity(N_, N_)+ gammaR_*Ppuk.transpose() * Ppuk;
    Hessian.block(3*N_+2*m_, 3*N_+2*m_, N_, N_) =  alphaR_*Matrix_t::Identity(N_, N_)+  betaR_*Ppu.transpose() * Ppu;
    g += Hessian.transpose() * u_0_;
    return g;
}

Matrix_t EssentialNonLinearSimulated::Getlb()
{
    // No constraints, return empty vector
    Matrix_t lb;
    return lb;
}

Matrix_t EssentialNonLinearSimulated::Getub()
{
    // No constraints, return empty vector
    Matrix_t ub;
    return ub;
}