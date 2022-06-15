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



#ifndef VISION_UTILS_H 
#define VISION_UTILS_H

#include <vector>
#include <random>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include "common.h"

enum class homographySolution {
    SOLUTION_1=0,
    SOLUTION_2=1,
    NONE=-1
};

Matrix_t ComputeHomography( const std::vector<cv::Point2d> &reference,
                            const std::vector<cv::Point2d> &current,
                            const Matrix_t &K);

Matrix_t ComputeEssential( const std::vector<cv::Point2d> &reference,
                           const std::vector<cv::Point2d> &current,
                           const Matrix_t &K);

void RecoverFromHomography(const Matrix_t &homography,Matrix_t &R, Vector3D_t &t, Vector3D_t &n, FPTYPE &sc, const int current_iteration, homographySolution &homograpy_solution);

void RecoverFromEssential(const Matrix_t &essential,Matrix_t &R, Vector3D_t &t);

void OpencvVector(const std::vector<Point2D_t> &reference,const std::vector<Point2D_t> &current,
                  const std::vector<bool> &visible,
                  std::vector<cv::Point2d> &cv_reference_points, std::vector<cv::Point2d> &cv_current_points);

class Camera {

    public:
        Camera(const FPTYPE &fx=1.0,const FPTYPE &fy=1.0,const FPTYPE &u0=0.0,const FPTYPE &uv=0.0,const FPTYPE& sigmaNoise=0.0);
        void UpdateCameraPosition(const Matrix_t &new_world_camera_position);
        void UpdateCameraOrientation();
        Point2D_t TakePictureImageCoordinates(const Point3D_t &points, bool addNoise=false) const;
        Point2D_t TakePictureCameraCoordinates(const Point3D_t &points) const;
        inline const Matrix_t &GetIntrisicParametersMatrix() const { return K_;};
        inline const Matrix_t &GetInverseIntrisicParametersMatrix() const { return Kinv_;};

    private:
        Matrix_t K_;
        Matrix_t Kinv_;
        AffineTransformation camera_T_world_;
        FPTYPE sigmaNoise_;
};


class InterestPoints {

    public:
        enum class OcclusionPolicy { None=0, FirstHalfOccluded=1, SecondHalfOccluded=2, RandomlyOccluded=3, TotalOcclusion=4};

        InterestPoints(
            Camera& camera,
            const std::vector<Point3D_t>& reference_points,
            const std::vector<Point3D_t>& world_points,
            bool addNoise=false,
            const OcclusionPolicy &policy=OcclusionPolicy::None, 
            const double &OcclusionProportion=0.0);

        InterestPoints(
            const std::vector<Point2D_t>& current_image_points,
            const std::vector<Point2D_t>& reference_image_points) : current_image_points_(current_image_points), reference_image_points_(reference_image_points), 
                                                                    current_visibility_(true,current_image_points.size()), occlusion_policy_(OcclusionPolicy::None), occlusion_proportion_(0.0) {}

        inline const std::vector<Point3D_t> &GetInCurrentWorldCoordinates() const {
            return current_world_points_;
        }
        inline const std::vector<Point2D_t> &GetInCurrentImageCoordinates() const {
            return current_image_points_;
        }
        inline const std::vector<Point3D_t> &GetInReferenceWorldCoordinates() const {
            return reference_world_points_;
        }
        inline const std::vector<Point2D_t> &GetInReferenceProjectiveCoordinates() const {
            return reference_projective_points_;
        }
        inline const std::vector<Point2D_t> &GetInReferenceImageCoordinates() const {
            return reference_image_points_;
        }
        inline const std::vector<bool> &GetVisibility() const {
            return current_visibility_;
        }
        inline const std::vector<Point2D_t> &SimulatedProjection(const Camera& camera, bool addNoise=false) {
            // Get the current interest points expressed in the world frame (this is possible only in this simulation class)
            for (int k=0; k<current_world_points_.size();k++) {
                // If the point is seen by the camera, keep it in current_image_points
                current_image_points_[k] = camera.TakePictureImageCoordinates(current_world_points_[k],addNoise);
                current_visibility_[k]   = true;
            }
            switch (occlusion_policy_) {
                case OcclusionPolicy::FirstHalfOccluded: 
                    for (int k=0;k<current_world_points_.size()/2;k++) {
                        current_visibility_[k]   = false;
                    }
                    break;
                case OcclusionPolicy::SecondHalfOccluded:
                    for (int k=current_world_points_.size()/2;k<current_world_points_.size();k++) {
                        current_visibility_[k]   = false;
                    }
                    break;
                case OcclusionPolicy::RandomlyOccluded: {
                        std::random_device rd;
                        std::mt19937 generator(rd());
                        std::bernoulli_distribution bernoulli(1.0-occlusion_proportion_);
                        for (int k=current_world_points_.size()/2;k<current_world_points_.size();k++) {
                            current_visibility_[k]   = bernoulli(generator);
                        }
                    }
                    break;
                case OcclusionPolicy::TotalOcclusion:{
                    for (int k=0;k<current_world_points_.size();k++) {
                        current_visibility_[k]   = false;
                    }
                }
                break;
                case OcclusionPolicy::None:             
                default:
                    // Let the visibility flag untouched
                    break;    
            }
            return current_image_points_;
        }
        inline void SetOcclusionPolicy(const OcclusionPolicy &policy,const double &OcclusionProportion=0.0) {
            occlusion_policy_     = policy;
            occlusion_proportion_ = OcclusionProportion;
        }
        void UpdateCurrentWorldCoordinates(const AffineTransformation &transformation);
        inline void UpdateCurrentImageCoordinates(const std::vector<Point2D_t> &newcurrent) {
        }

    private:
        OcclusionPolicy occlusion_policy_;
        double occlusion_proportion_;
        std::vector<Point3D_t> current_world_points_;
        std::vector<Point2D_t> current_image_points_;
        std::vector<bool> current_visibility_;        
        std::vector<Point3D_t> reference_world_points_;
        std::vector<Point2D_t> reference_projective_points_;
        std::vector<Point2D_t> reference_image_points_;
};

#endif // VISION_UTILS_H
