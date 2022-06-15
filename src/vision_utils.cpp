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

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include <opencv2/opencv.hpp>

#include "vision_utils.h"
#include "common.h"


/*******************************************************************************
 *
 *                                functions
 *
 ******************************************************************************/


namespace {


inline std::pair<FPTYPE, FPTYPE> unitize(FPTYPE x, FPTYPE y)
{
    FPTYPE magnitude = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
    FPTYPE l = x / magnitude;
    FPTYPE m = y / magnitude;
    return std::make_pair(l, m);
}

Vector3D_t vex(const Matrix_t &A) {
    if (A.rows()!=3 || A.cols()!=3)
        throw std::domain_error("vex: expects 3x3 matrices as an input");
    Vector3D_t v;
    v(0)=0.5*(A(2,1)-A(1,2));
    v(1)=0.5*(A(0,2)-A(2,0));
    v(2)=0.5*(A(1,0)-A(0,1));
    return v;
}

Matrix_t GetRotationMean(const Matrix_t &R1, const Matrix_t &R2)
{
    Matrix_t tmp = R1.transpose() * R2;
    return R1 * tmp.sqrt();
}

}; // namespace

void OpencvVector(const std::vector<Point2D_t> &reference, const std::vector<Point2D_t> &current,
                  const std::vector<bool> &visible, std::vector<cv::Point2d> &cv_reference_points,
                  std::vector<cv::Point2d> &cv_current_points) {
    // It is assumed both vectors have the same number of elements
    if (!visible.size())
        for (int i = 0; i < reference.size(); i++) {
            cv_reference_points.push_back(cv::Point2d(X(reference[i]), Y(reference[i])));
            cv_current_points.push_back(cv::Point2d(X(current[i]), Y(current[i])));
        }
    else {
        for (int i = 0; i < reference.size(); i++) if (visible[i]) {
                cv_reference_points.push_back(cv::Point2d(X(reference[i]), Y(reference[i])));
                cv_current_points.push_back(cv::Point2d(X(current[i]), Y(current[i])));
            }
    }

}

Matrix_t ComputeHomography( const std::vector<cv::Point2d> &cv_reference_points,
                            const std::vector<cv::Point2d> &cv_current_points,
                            const Matrix_t &K)
{
    cv::Mat cv_homography = cv::findHomography(cv_reference_points, cv_current_points);
    Matrix_t homography(3, 3);
    for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
            homography(i,j) = cv_homography.at<double>(i,j);
    return K.inverse()*homography*K;
}

Matrix_t ComputeEssential( const std::vector<cv::Point2d> &cv_reference_points,
                           const std::vector<cv::Point2d> &cv_current_points,
                           const Matrix_t &K)
{
    cv::Mat cv_fundamental_matrix =
        findFundamentalMat(cv_reference_points, cv_current_points, cv::FM_8POINT, 3, 0.99);
    Matrix_t fundamental(3, 3);
    for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
            fundamental(i,j) = cv_fundamental_matrix.at<double>(i,j);
    return K.transpose()*fundamental*K;
}

void RecoverFromEssential(const Matrix_t &homography,Matrix_t &R, Vector3D_t &t)
{
    // Decomposes the essential matrix E (3x3) into the camera motion
    // In practice there are multiple solutions and S (4x4xN) is a set of homogeneous
    // transformations representing possible camera motion.
    //
    // Reference::
    //
    // Y.Ma, J.Kosecka, S.Soatto, S.Sastry,
    // "An invitation to 3D",
    // Springer, 2003.
    // p116, p120-122
    // Notes::
    // - The transformation is from view 1 to view 2.
    // See also CentralCamera.E.
    // we return T from view 1 to view 2
    Matrix_t U;
    Matrix_t S;
    Matrix_t V;

    Eigen::JacobiSVD<Matrix_t> svd(homography, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    S = svd.singularValues();
    V = svd.matrixV();
    // Ma etal solution, p116, p120-122
    // Fig 5.2 (p113), is wrong, (R,t) is from camera 2 to 1
    if (V.determinant() < 0) {
        V = -V;
        S = -S;
    }
    if (U.determinant() < 0) {
        U = -U;
        S = -S;
    }
    static Eigen::Matrix3d mp(Eigen::AngleAxisd(+0.5*M_PI, Eigen::Vector3d::UnitZ()));
    static Eigen::Matrix3d mm(Eigen::AngleAxisd(-0.5*M_PI, Eigen::Vector3d::UnitZ()));
    Matrix_t R1 = U*mp.transpose()*V.transpose();
    Matrix_t R2 = U*mm.transpose()*V.transpose();
    Vector3D_t t1 = vex(U*mp*S.asDiagonal()*U.transpose());
    Vector3D_t t2 = vex(U*mm*S.asDiagonal()*U.transpose());   
    if (R1(0,0)>0&&R1(1,1)>0&&R1(2,2)>0) 
       R=R1;
    else
       R=R2;
    if (t1(2)>0) // Suppose here that the target camera position is in front of us, not behind 
       t=t1;
    else
       t=t2;
}

void RecoverFromHomography(const Matrix_t &homography,Matrix_t &R, Vector3D_t &t, Vector3D_t &n, FPTYPE &sc, const int current_iteration, homographySolution &homograpy_solution)
{
    Matrix_t U;
    Matrix_t S;
    Matrix_t V;

    {
        Eigen::JacobiSVD<Matrix_t> svd(homography, Eigen::ComputeFullU | Eigen::ComputeFullV);
        U = svd.matrixU();
        S = svd.singularValues();
        V = svd.matrixV();
    }

    // Eigen returns the transpose of what we need here
    V.transposeInPlace();

    FPTYPE s1 = S(0) / S(1);
    FPTYPE s3 = S(2) / S(1);
    FPTYPE zeta = s1 - s3;
    if (fabs(zeta)<std::numeric_limits<FPTYPE>::epsilon()) {
        sc = 1.0;
        R = Matrix_t::Identity(3,3);
        n = Vector3D_t::Zero(3); n(0)=1.0;
        t = Vector3D_t::Zero(3); t(0)=1.0;
        return;
    }
    FPTYPE a1 = std::sqrt(1 - std::pow(s3, 2));
    FPTYPE b1 = std::sqrt(std::pow(s1, 2) - 1);
    std::pair<FPTYPE,FPTYPE> p,q,r;
    p = unitize(a1, b1);               FPTYPE &a = p.first;  FPTYPE &b = p.second;
    q = unitize(1 + s1 * s3, a1 * b1); FPTYPE &c = q.first;  FPTYPE &d = q.second;
    r = unitize(-b / s1, -a / s3);     FPTYPE &e = r.first;  FPTYPE &f = r.second;

    Matrix_t v1 = V.row(0).transpose();
    Matrix_t v3 = V.row(2).transpose();

    Vector3D_t n1 = b * v1 - a * v3;
    Vector3D_t n2 = b * v1 + a * v3;

    Matrix_t tmp(3, 3);
    tmp << c, 0.0, d,
           0, 1.0, 0,
          -d, 0.0, c;

    Matrix_t R1 = U * (tmp * V);
    Matrix_t R2 = U * (tmp.transpose() * V);

    Vector3D_t t1 = e * v1 + f * v3;
    Vector3D_t t2 = e * v1 - f * v3;

    if (Z(n1) < 0) {
        t1 = -t1;
        n1 = -n1;
    }

    if (Z(n2) < 0) {
        t2 = -t2;
        n2 = -n2;
    }
    if(current_iteration == 0) {
        if (Z(n1) > Z(n2)) { //Solution 1
            R = R1.transpose();
            t = zeta * t1;
            n = n1;
            homograpy_solution = homographySolution::SOLUTION_1;
        } else { //Solution 2
            R = R2.transpose();
            t = zeta * t2;
            n = n2;
            homograpy_solution = homographySolution::SOLUTION_2;
        }
    }else{
        if(homograpy_solution == homographySolution::SOLUTION_1){
            R = R1.transpose();
            t = zeta * t1;
            n = n1;
        }else{
            R = R2.transpose();
            t = zeta * t2;
            n = n2;
        }
    }

    sc = 1.0 / zeta;
}


/*******************************************************************************
 *
 *                                Camera
 *
 ******************************************************************************/

Camera::Camera(const FPTYPE &fx, const FPTYPE &fy, const FPTYPE &u0, const FPTYPE &v0, const FPTYPE &sigmaNoise)
{
    Matrix_t K(3,3);
    K << fx ,0.0, u0,
        0.0, fy, v0,
        0.0,0.0,1.0;
    // intrinsic parameters
    K_ = K;
    Kinv_= K.inverse();
    sigmaNoise_ = sigmaNoise;
    // extrinsic parameters
    Eigen::AngleAxis<FPTYPE> camera_orientation;

    camera_T_world_ = Eigen::Translation<FPTYPE, 3>(0.0, 0.0, 0.0) *
                      Eigen::AngleAxis<FPTYPE>(M_PI_2, Point3D_t::UnitX()) *
                      Eigen::AngleAxis<FPTYPE>(M_PI_2, Point3D_t::UnitZ());
}

void Camera::UpdateCameraPosition(const Matrix_t &new_world_camera_position)
{
    Matrix_t camera_R_world = camera_T_world_.rotation();
    camera_T_world_.translation() = -camera_R_world*new_world_camera_position;
}


void Camera::UpdateCameraOrientation()
{
    throw std::exception();
}


Point2D_t Camera::TakePictureImageCoordinates(const Point3D_t &world_points, bool addNoise) const
{
    Point2D_t points_in_camera_coordinates = TakePictureCameraCoordinates(world_points);
    // Come back to homogeneous
    Point3D_t picture                     = points_in_camera_coordinates.colwise().homogeneous().matrix();
    Point2D_t points_in_image_coordinates = (K_*picture).colwise().hnormalized();
    if (addNoise) {
        // Add noise
        cv::Mat noise  = cv::Mat::zeros(2,1,CV_64FC1);
        cv::randn(noise,  0.0, sigmaNoise_);
        points_in_image_coordinates(0) += noise.at<double>(0,0);
        points_in_image_coordinates(1) += noise.at<double>(1,0);
    }
    return points_in_image_coordinates;
}


Point2D_t Camera::TakePictureCameraCoordinates(const Point3D_t &world_points) const
{
    Point3D_t picture;
    // World frame point moved to camera frame 
    picture = camera_T_world_ * world_points.colwise().homogeneous();
    // Normalized by the third coordinate
    return picture.colwise().hnormalized();
}


/*******************************************************************************
 *
 *                             InterestPoints
 *
 ******************************************************************************/

InterestPoints::InterestPoints(
    Camera& camera,
    const std::vector<Point3D_t>& reference_points,
    const std::vector<Point3D_t>& current_world_points,
    bool addNoise,
    const OcclusionPolicy &policy, 
    const double &OcclusionProportion)
    : current_world_points_(current_world_points)
    , reference_world_points_(reference_points)
    , current_visibility_(true,current_world_points.size())
    , occlusion_policy_(policy), occlusion_proportion_(OcclusionProportion)
{
    for (const auto& point : reference_world_points_) {
        reference_projective_points_.push_back(camera.TakePictureCameraCoordinates(point));
        reference_image_points_.push_back(camera.TakePictureImageCoordinates(point,addNoise));
        current_image_points_.push_back(Point2D_t(0,0));
    }
}

void InterestPoints::UpdateCurrentWorldCoordinates(const AffineTransformation &transformation)
{
    std::vector<Point3D_t> new_world_points;
    for (const auto &points : current_world_points_)
        new_world_points.push_back(transformation * points);
    current_world_points_ = new_world_points;
}
