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

#ifndef FORMULATIONS_H_
#define FORMULATIONS_H_

#include <string>
#include <map>
#include <queue>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "common_formulation.h"
#include "vision_utils.h"
#include "trunk_orientation.h"

/*******************************************************************************
 *
 *                                   HerdtBase
 *
 ******************************************************************************/

class HerdBase : public FormulationBase, public SimulatorInterface
{
    public:
    HerdBase(const boost::property_tree::ptree &parameters);
    HerdBase(const std::string nameFileParameters);
    ~HerdBase() {};
    virtual Matrix_t GetH() override;

    void Update(const Matrix_t &solution) override;
        void LogCurrentResults(const Matrix_t &solution=Matrix_t()) const override;       
        void LogCurrentPredictions(const Matrix_t &solution=Matrix_t()) const override;               
        inline bool Continue() const override {
            return current_iteration_ < iterations_;
        }
        void LogConfigurationDetails() const;
        inline Point3D_t GetCoMAbsolutePosition() const override {
            // Position of the CoM: zero
            Point3D_t point(0.0,0.0,0.0);
            // The positions are mapped to the world frame for proper display
            point = world_T_com_ * point;
            return point;
        };     
        inline int GetCurrentStepRemainingCycles() const override {
            return step_generator_.GetRemainingSteps();
        };     
        inline bool GetCurrentSupportType() const override {
            return (step_generator_.GetSupportPhase()==SupportPhase::SINGLE_SUPPORT);
        };     
        inline FPTYPE GetCurrentStepRatio() const override {
            if (step_generator_.GetSupportPhase()==SupportPhase::SINGLE_SUPPORT)
                return 1.0-(step_generator_.GetRemainingSteps()-1)/((FPTYPE)step_generator_.GetSingleSupportLen()-1.0);
            else
                return 1.0-(step_generator_.GetRemainingSteps()-1)/((FPTYPE)step_generator_.GetDoubleSupportLen()-1.0);
        }; 
        bool GetSupportFootId() const override { return step_generator_.GetSupportFoot()==Foot::LEFT_FOOT;};
        // TODO: IN THIS CASE?
        inline FPTYPE GetCoMAbsoluteTheta() const override { return world_com_orientation_;};       
        inline Point3D_t GetSupportFootAbsolutePosition() const override {
            // Get the support foot position
            Point3D_t point(support_foot_position_(0), support_foot_position_(1), -robot_physical_parameters_->com_height());
            // Maps it into the world frame            
            return world_T_com_ * point;
        };   
        // TODO: IN THIS CASE?
        inline FPTYPE GetSupportFootAbsoluteTheta() const override {return world_support_foot_orientation_;};     
        // TODO: IN THIS CASE?         
        inline FPTYPE GetSwingFootAbsoluteTheta() const override {return world_flying_foot_orientation_;};   
        // TODO: IN THIS CASE?
        inline FPTYPE GetNextSupportFootAbsoluteTheta() const override {return world_next_support_foot_orientation_;}; 
        inline void SetCoMValues(const FPTYPE x_position, const FPTYPE y_position, const FPTYPE z_position, const FPTYPE x_speed, const FPTYPE y_speed )  override{
            // Position of the CoM in the world
            Point3D_t point(x_position,y_position,z_position);
            // The positions are mapped to the world frame for proper display
            Point3D_t com_p_world = world_T_com_.inverse() * point;
            dynamic_model_.SetCoM_X_Position(com_p_world(0,0));
            dynamic_model_.SetCoM_Y_Position(com_p_world(1,0));
            dynamic_model_.SetCoM_X_Speed(x_speed);
            dynamic_model_.SetCoM_Y_Speed(y_speed);
        };
        inline Point3D_t GetNextSupportFootAbsolutePosition(const Matrix_t &solution) const override {
            // Get the first footstep final position
            Point3D_t point(solution(N_, 0), solution(N_ + m_ + N_, 0), -robot_physical_parameters_->com_height());
            // Maps it into the world frame
            return world_T_com_ * point;
        }; 
        inline bool IsSupportFootChange() const  override {return !step_generator_.IsSameSupportFoot();}


        virtual void PrepareDataForNextOptimization();
        virtual void UpdateSimulation(const Matrix_t &solution=Matrix_t());

        

    protected:
        virtual void SolveOrientation() = 0;

        FPTYPE alpha_;
        FPTYPE beta_;
        FPTYPE gamma_; 
        
        FPTYPE world_next_support_foot_orientation_;
        // Global orientations (for displaying results)
        FPTYPE world_support_foot_orientation_;
        FPTYPE world_flying_foot_orientation_;
        FPTYPE world_com_orientation_;

        AffineTransformation world_T_foot_;
        AffineTransformation com_T_foot_;
        AffineTransformation world_T_next_foot_;
        AffineTransformation com_T_next_foot_;
        // Transformation that maps point from
        AffineTransformation mk_T_mk1;

        Eigen::Transform<FPTYPE, 3, Eigen::Affine> world_T_com_;
        
        int iterations_;
};

//******************************************************************************
//
//                        QPHerdtSim
//
//******************************************************************************

class Herdt : public HerdBase 
{

    public:
        Herdt(const boost::property_tree::ptree &parameters);
        Herdt(const std::string nameFileParameters);
        ~Herdt() { };

         Matrix_t Getg() override;
         
    protected:
        void SolveOrientation() override;
        std::vector<FPTYPE> reference_com_x_speed_;
        std::vector<FPTYPE> reference_com_y_speed_;
        std::vector<FPTYPE> reference_angles_;  

};

//******************************************************************************
//
//                        QPHerdtReal
//
//******************************************************************************

class HerdtReal : public HerdBase
{
    public:
        HerdtReal(const boost::property_tree::ptree &parameters, const FPTYPE x_speed_ref, const FPTYPE y_speed_ref, const FPTYPE theta_ref );
        HerdtReal(const std::string nameFileParameters, const FPTYPE x_speed_ref, const FPTYPE y_speed_ref, const FPTYPE theta_ref );
        void SetCurrentReferenceSpeed(const FPTYPE x_speed_ref, const FPTYPE y_speed_ref, const FPTYPE theta_ref);
        ~HerdtReal() { };

        Matrix_t Getg() override;

    protected:
        void SolveOrientation() override;

    private:
        FPTYPE x_speed_ref_;
        FPTYPE y_speed_ref_;
        FPTYPE theta_ref_;
        
};


/*******************************************************************************
 *
 *                                   Homography
 *
 ******************************************************************************/

struct VisualFeatureData {
    VisualFeatureData(const std::string &name, int N, bool linear=true) { 
        if (linear) {
    	    if (name.size() && name[0]=='h')
                model = boost::shared_ptr<MPCVisualConstraint>(new MPCLinearHomography(name,N));
            else if (name.size() && name[0]=='e')  	
                model = boost::shared_ptr<MPCVisualConstraint>(new MPCLinearEssential(name,N));	 
            else
                throw std::exception();
        } else {
            if (name.size() && name[0]=='e')   
                model = boost::shared_ptr<MPCVisualConstraint>(new MPCNonLinearEssential(name,N));  
            else if (name.size() && name[0]=='h')
                model = boost::shared_ptr<MPCVisualConstraint>(new MPCNonLinearHomography(name,N));
            else
                throw std::exception();
        }
    };
    boost::shared_ptr<MPCVisualConstraint> model;
    FPTYPE predicted;
    FPTYPE actual;
};

class VisualFeatureBase : public FormulationBase, public SimulatorInterface {
    public:
        enum class MultipleObjectives { WeightedAverages=0, SharedPredictionWindows=1};
        VisualFeatureBase(const boost::property_tree::ptree &parameters);
        VisualFeatureBase(const std::string nameFileParameters);
        ~VisualFeatureBase() { };
        virtual void Update(const Matrix_t &solution) override;
        void LogCurrentResults(const Matrix_t &solution=Matrix_t()) const override;
        void LogCurrentPredictions(const Matrix_t &solution=Matrix_t()) const override;        
        inline bool Continue() const override {
            return current_iteration_ < iterations_;
        };
        inline Point3D_t GetCoMAbsolutePosition() const override {
            // Position of the CoM: zero
            Point3D_t point(0.0,0.0,0.0);
            // The positions are mapped to the world frame for proper display
            return world_T_com_ * point;
        };   
        inline FPTYPE GetCoMAbsoluteTheta() const override { return world_com_orientation_;};   
        inline int GetCurrentStepRemainingCycles() const override {
            return step_generator_.GetRemainingSteps();
        };     
        inline bool GetCurrentSupportType() const override {
            return (step_generator_.GetSupportPhase()==SupportPhase::SINGLE_SUPPORT);
        };     
        inline FPTYPE GetCurrentStepRatio() const override {
            if (step_generator_.GetSupportPhase()==SupportPhase::SINGLE_SUPPORT)
                return (step_generator_.GetSingleSupportLen()-step_generator_.GetRemainingSteps())/((FPTYPE)step_generator_.GetSingleSupportLen());
            else {
                return (step_generator_.GetDoubleSupportLen()-step_generator_.GetRemainingSteps())/((FPTYPE)step_generator_.GetDoubleSupportLen());
            }
        }; 

        bool GetSupportFootId() const override { return step_generator_.GetSupportFoot()==Foot::RIGHT_FOOT;};
        inline Point3D_t GetSupportFootAbsolutePosition() const override {
            Point3D_t point(support_foot_position_(0), support_foot_position_(1), -robot_physical_parameters_->com_height());
            return world_T_com_ * point;
        };   
        inline FPTYPE GetSupportFootAbsoluteTheta() const override {return world_support_foot_orientation_;};      
        inline FPTYPE GetSwingFootAbsoluteTheta() const override {return world_flying_foot_orientation_;}; 
        inline FPTYPE GetNextSupportFootAbsoluteTheta()  const override {return world_next_support_foot_orientation_;};   
        inline Point3D_t GetNextSupportFootAbsolutePosition(const Matrix_t &solution) const override {
            // Get the first footstep final position
            Point3D_t point(solution(N_, 0), solution(N_ + m_ + N_, 0), -robot_physical_parameters_->com_height());
            // Maps it into the world frame
            return world_T_com_ * point;
        };
        inline bool IsSupportFootChange() const  override {return !step_generator_.IsSameSupportFoot();}

        inline void SetCoMValues(const FPTYPE x_position, const FPTYPE y_position, const FPTYPE z_position, const FPTYPE x_speed, const FPTYPE y_speed ) override{

            // Position of the CoM: zero
            Point3D_t point(x_position,y_position,z_position);
            // The positions are mapped to the world frame for proper display
            Point3D_t com_p_world = world_T_com_.inverse() * point;
            dynamic_model_.SetCoM_X_Position(com_p_world(0,0));
            dynamic_model_.SetCoM_Y_Position(com_p_world(1,0));
            dynamic_model_.SetCoM_X_Speed(x_speed);
            dynamic_model_.SetCoM_Y_Speed(y_speed);
        };
    protected:
        FPTYPE alpha_;
        FPTYPE beta_;
        FPTYPE gamma_;
        FPTYPE eta_x_;
        FPTYPE eta_y_;
        FPTYPE kappa_;
        FPTYPE x_speed_ref_;
        int min_points_to_estimate_constraint_;
        int n_simulated_points_;
        int iterations_;
        int first_reference_image_;
        int last_reference_image_;
        int occlusion_start_;
        int occlusion_end_;
        InterestPoints::OcclusionPolicy occlusion_policy_;
        FPTYPE occlusion_proportion_;
        const int REFERENCE_WINDOW_ = 2;
        MultipleObjectives multiple_objective_method_;
        bool u_0_prev;

        // Transform from CoM to world. Used to plot.
        AffineTransformation world_T_com_;
        AffineTransformation camera_T_com_;
        AffineTransformation world_T_foot_;
        AffineTransformation com_T_foot_;
        AffineTransformation world_T_next_foot_;
        AffineTransformation com_T_next_foot_;

        // Names of the visual features to be used
        std::vector<std::string> all_visual_features_names;

        // Vector (for each reference image) of vectors of visual features
        std::vector<std::vector<VisualFeatureData> > all_visual_features_;

        // The following structures are used in the case of simulated scenarios
        // They contain the 3D point information
        std::vector<InterestPoints> all_interest_points_;
        std::vector<InterestPoints> all_ground_truth_interest_points_;
        std::vector<InterestPoints> all_virtual_interest_points_;
        std::vector<InterestPoints> all_ground_truth_virtual_interest_points_;
        std::map<std::string, FPTYPE> expected_values;	

        // TODO: rename this to meaningful
        std::map<std::string, FPTYPE> latest_visual_data_;

        // Counter to compute the rolling average of errors
        unsigned int number_of_samples_;

        // Gains associated to each visual feature
        std::vector<FPTYPE> error_gains_;

        // Perturbations
        bool push_com_;
        FPTYPE push_com_intensity_;
        int push_com_iteration_;
    
        bool instant_switch_;
        FPTYPE switch_threshold_;
        bool index_switch_threshold_;
        
        FPTYPE d_;
        FPTYPE c_x_com_;
        FPTYPE c_z_com_;
        Camera camera_;
        Camera camera_ground_truth_;

        // Global orientations (for displaying results)
        FPTYPE world_support_foot_orientation_;
        FPTYPE world_flying_foot_orientation_;
        FPTYPE world_com_orientation_;
        FPTYPE world_next_support_foot_orientation_;

        // Transformation that maps point from
        AffineTransformation mk_T_mk1;

        // Initial value in the case of a non-linear formulation
        Matrix_t u_0_;

        // TODO: CHECK IF OK HERE? Could the dynamical model be used instead?
        Matrix_t thetaRef_;

        //For the flyin foot
        Matrix_t theta_ref_foot_;

        std::vector<FPTYPE> thetasFoot_;
        const FPTYPE SOFT_ANGLE =  2.5 * M_PI / 180.0;


    protected:
        virtual void SolveOrientation() = 0;
        virtual void UpdatePredictedValues(const Matrix_t &solution) {};
        virtual void PrepareDataForNextOptimization(const Matrix_t &solution=Matrix_t());
        virtual void UpdateSimulation(const Matrix_t &solution=Matrix_t());
        virtual void SimulateVisualFeatures() = 0;
        virtual void SetNewU_0(const Matrix_t &u_0, const bool change_foot) = 0;
        virtual void UpdateActualVisualData() = 0;
        void UpdateReferenceImages_();
        void TakePictureReferenceFromCurrentPosition();
        std::vector<FPTYPE> GetReferenceWeights();
	    std::vector<Point3D_t> Load3DReferencePoints_(
            const boost::property_tree::ptree& parameters,
            const AffineTransformation& com_T_world,
            const int targetId);
        void InitOptionalParameters(const boost::property_tree::ptree& parameters);
        void InitReferenceImage_(
            const boost::property_tree::ptree& parameters,
            const std::vector<Point3D_t> &reference_points_in_world_coordinates,
            const int id,
            bool virtualImage,
            Camera camera_copy,
            bool groundTruth);
        void RefAnglesInterpolation(FPTYPE ref_angle);
        Matrix_t nlGetA();
        Matrix_t nlGetlbA();
        Matrix_t nlGetubA();

        virtual int GetIndexl(FPTYPE hr, const Matrix_t &Hp) = 0;
        virtual Matrix_t GetDs(std::vector<VisualFeatureData> &current_visual_features) = 0;
};

class HomographyBase : public VisualFeatureBase {
    public:
        HomographyBase(const boost::property_tree::ptree &parameters);
        HomographyBase(const std::string nameFileParameters);
        Matrix_t GetH() override;
        Matrix_t Getg() override;
    protected:    
        std::vector<Matrix_t> homography_matrices_;
        std::vector<Matrix_t> homography_matrices_ground_truth_;
        virtual void UpdatePredictedValues(const Matrix_t &solution) override;
        virtual void SimulateVisualFeatures() override;
        void UpdateActualVisualData() override;
        void SolveOrientation() override;
        FPTYPE DistanceFromPlane(const Vector3D_t &n, FPTYPE dInit, FPTYPE xk, FPTYPE yk);
        void SetNewU_0(const Matrix_t &u_0, const bool change_foot) override {
            // jerks_x (shift by one)
            for(int i=0;i<=(N_-2);i++){
                u_0_(i,0) = u_0(i+1,0);
            }
            u_0_(N_-1,0) = u_0(N_-1,0);

            // Foot_position_x
            if(change_foot){
                u_0_(N_,0)   = u_0(N_+1,0);
                u_0_(N_+1,0) = u_0(N_,0) + u_0(N_+1,0);
            } else{
                for(int i=N_;i<(m_+N_);i++){
                    u_0_(i,0) = u_0(i,0);
                }
            }
            // jerks_y (shift by one)
            for(int i=N_+m_;i<=(2*N_+m_-2);i++){
                u_0_(i,0) = u_0(i+1,0);
            }
            u_0_(2*N_+m_-1,0) = u_0(2*N_+m_-1,0);

            // Foot_position_y
            if(change_foot){
                u_0_(2*N_+m_,0)   = u_0(2*N_+m_+1,0);
                u_0_(2*N_+m_+1,0) = u_0(2*N_+m_,0) + u_0(2*N_+m_+1,0);
            } else{
                for(int i=2*N_+m_;i<(2*N_+2*m_);i++){
                    u_0_(i,0) = u_0(i,0);
                }
            }

            // jerks_foot (shift by one)
            for(int i=2*N_+2*m_;i<=(3*N_+2*m_-2);i++){
                u_0_(i,0) =  u_0(i+1,0);
            }
            u_0_(3*N_+2*m_-1,0) = u_0(3*N_+2*m_-1,0);

            // jerks_com (shift by one)
            for(int i=3*N_+2*m_;i<=(4*N_+2*m_-2);i++) {
                u_0_(i,0) = 0.0; //u_0(i+1,0);
            }
            u_0_(4*N_+2*m_-1,0) = 0.0;// u_0(4*N_+2*m_-1,0);

        }

        homographySolution  homography_solution_;

        Matrix_t GetDs(std::vector<VisualFeatureData> &current_visual_features) override ;
        int GetIndexl(FPTYPE hr, const Matrix_t &Hp) override {
            for(int i=0;i<N_;i++){
                if( std::abs(Hp(i,0)-hr)<switch_threshold_){
                    return i;
                }
            }
            return  N_;
        }
};

class Homography : public HomographyBase {
    public:
        Homography(const boost::property_tree::ptree &parameters,
                   const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                   const FPTYPE h21, const FPTYPE h22, const FPTYPE h23,
                   const FPTYPE h31, const FPTYPE h32, const FPTYPE h33 );
        Homography(const std::string nameFileParameters,
                   const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                   const FPTYPE h21, const FPTYPE h22, const FPTYPE h23,
                   const FPTYPE h31, const FPTYPE h32, const FPTYPE h33 ); // To python interface
        void SetCurrentHomography( const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                                   const FPTYPE h21, const FPTYPE h22, const FPTYPE h23,
                                   const FPTYPE h31, const FPTYPE h32, const FPTYPE h33, bool isComputeHomography);
        void UpdateSimulation(const Matrix_t &solution=Matrix_t()) override;

};

class HomographyObstacles : public Homography {
    public:
        HomographyObstacles(const boost::property_tree::ptree &parameters,
                            const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                            const FPTYPE h21, const FPTYPE h23,
                            const FPTYPE h31, const FPTYPE h32, const FPTYPE h33, 
                            const FPTYPE c1,  const FPTYPE c2,  const FPTYPE c3, const bool isObstacle);
        HomographyObstacles(const std::string nameFileParameters,
                            const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                            const FPTYPE h21, const FPTYPE h23,
                            const FPTYPE h31, const FPTYPE h32, const FPTYPE h33, 
                            const FPTYPE c1,  const FPTYPE c2,  const FPTYPE c3, const bool isObstacle); // To python interface

        void SetCurrentHomography( const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                                   const FPTYPE h21, const FPTYPE h23,
                                   const FPTYPE h31, const FPTYPE h32, const FPTYPE h33, const bool isObstacle, const bool isComputeHomography,
                                   const FPTYPE c1,  const FPTYPE c2,  const FPTYPE c3, const int iteration);

        int GetNumberOfConstraints() override {return 5 * N_ + 4 * m_ ;};

    protected:
        Matrix_t GetA() override;
        Matrix_t GetubA() override;
        Matrix_t GetlbA() override;
        void SetC1AndC2(const FPTYPE c1,  const FPTYPE c2,  const int iteration, const bool isObstacle);
        Matrix_t GetC1();
        Matrix_t GetC2();
        void SetBobs(const FPTYPE c3, const int iteration, const bool isObstacle);
        Matrix_t GetBobs();
        Matrix_t ObstacleRestrictionsA(const Matrix_t &Ppu);
        Matrix_t ObstacleBoundaryVector(const Matrix_t &Pps, const Matrix_t &x_state, const Matrix_t &y_state);

    private:

        Matrix_t C1_;
        Matrix_t C2_;
        Matrix_t Bobs_;
    
};

class HomographySimulated : public HomographyBase {
    public:
        HomographySimulated(const boost::property_tree::ptree &parameters);
        HomographySimulated(const std::string nameFileParameters);
        ~HomographySimulated() { };
};

class HomographyNonLinearSimulated : public HomographySimulated {
    public:
        HomographyNonLinearSimulated(const boost::property_tree::ptree &parameters);
        ~HomographyNonLinearSimulated() { };
        Matrix_t GetH() override;
        Matrix_t Getg() override;
        Matrix_t Getlb() override;
        Matrix_t Getub() override;
        int GetNumberOfVariables() override {return 4*N_+2*m_;};
        int GetNumberOfConstraints() override {return 3 * N_ + 2 * m_;};
        
    protected:
        Matrix_t GetA() override {return nlGetA();};
        Matrix_t GetlbA() override {return nlGetlbA();};
        Matrix_t GetubA() override {return nlGetubA();};        

        FPTYPE alpha_R_trunk_;
        FPTYPE alpha_R_foot_;
        FPTYPE betaR_;
        FPTYPE gammaR_;
};

class EssentialBase : public VisualFeatureBase {
    public:
        EssentialBase(const boost::property_tree::ptree &parameters);
        virtual Matrix_t GetH() override;
        virtual Matrix_t Getg() override;
    protected:            
        std::vector<Matrix_t> essential_matrices_;
        std::vector<Matrix_t> essential_matrices_ground_truth_;
        virtual void UpdatePredictedValues(const Matrix_t &solution) override;
        virtual void SimulateVisualFeatures() override;
        void UpdateActualVisualData() override;
        void SolveOrientation() override;
        void SetNewU_0(const Matrix_t &u_0, const bool change_foot) override {
            // TODO: avoid the duplication of the same function as homography; put the method in visualfeature instead?
            // jerks_x
            for(int i=0;i<=(N_-2);i++){
                u_0_(i,0) = u_0(i+1,0);
            }
            u_0_(N_-1,0) = u_0(N_-1,0);

            // Foot_position_x
            // BUG(?): the shifting by one unit does not apply in general for the footsteps, except at support foot changes
            for(int i=N_;i<=(m_+N_);i++){
                u_0_(i,0) = u_0(i,0);
            }

            // jerks_y
            for(int i=N_+m_;i<=(2*N_+m_-2);i++){
                u_0_(i,0) = u_0(i+1,0);
            }
            u_0_(2*N_+m_-1,0) = u_0(2*N_+m_-1,0);

            // Foot_position_y
            // BUG(?): the shifting by one unit does not apply in general for the footsteps, except at support foot changes
            if(change_foot){
                for(int i=2*N_+m_;i<=(2*N_+2*m_);i++){
                    u_0_(i,0) = 0.0;//u_0(i,0);
                }
            } else{
                for(int i=2*N_+m_;i<=(2*N_+2*m_);i++){
                    u_0_(i,0) = u_0(i,0);
                }
            }

            // jerks_com
            for(int i=2*N_+2*m_;i<=(3*N_+2*m_-2);i++){
                u_0_(i,0) = u_0(i+1,0);
            }
            u_0_(3*N_+2*m_-1,0) = u_0(3*N_+2*m_-1,0);

            // jerks_foot
            for(int i=3*N_+2*m_;i<=(4*N_+2*m_-2);i++) {
                u_0_(i,0) = u_0(i+1,0);
            }
            u_0_(4*N_+2*m_-1,0) = u_0(4*N_+2*m_-1,0);
        }
        FPTYPE h;
        FPTYPE e11_predicted_;
        FPTYPE e13_predicted_;
        FPTYPE e22_predicted_;
        FPTYPE e31_predicted_;
        FPTYPE e33_predicted_;

        Matrix_t GetDs(std::vector<VisualFeatureData> &current_visual_features) override ;
        int GetIndexl(FPTYPE hr, const Matrix_t &Hp) override {
            for(int i=0;i<N_;i++){
                if( std::abs(Hp(i,0)-hr)<switch_threshold_)
                    return i;
            }
            return  N_;
        }
    };

class EssentialSimulated : public EssentialBase {
    public:
        EssentialSimulated(const boost::property_tree::ptree &parameters);
        ~EssentialSimulated() { };
};

class EssentialNonLinearSimulated : public EssentialSimulated {
    public:
        EssentialNonLinearSimulated(const boost::property_tree::ptree &parameters);
        ~EssentialNonLinearSimulated() { };
        Matrix_t GetH() override;
        Matrix_t Getg() override;
        Matrix_t Getlb() override;
        Matrix_t Getub() override;
        int GetNumberOfVariables() override {return 4*N_+2*m_;};
        int GetNumberOfConstraints() override {return 3 * N_ + 2 * m_;};
        
    protected:
        Matrix_t GetA() override {return nlGetA();};
        Matrix_t GetlbA() override {return nlGetlbA();};
        Matrix_t GetubA() override {return nlGetubA();};    
        FPTYPE alphaR_;
        FPTYPE betaR_;
        FPTYPE gammaR_;
};

class SimulatorFactory {
   public:
        SimulatorFactory();
        ~SimulatorFactory() {};
        boost::shared_ptr<SimulatorInterface> BuildSimulator(const boost::property_tree::ptree &parameters);
    private:
        void VerifyParameters(const std::vector<std::string> &all_required_parameters, 
                              const boost::property_tree::ptree &parameters);
        void LogInputParameters(const std::vector<std::string> &all_required_parameters, const boost::property_tree::ptree &parameters);
        std::map<std::string, std::vector<std::string> > requiredParameters_;
};
/*

Ejemplo de como se definiria una clase nueva para otra simluacion, hay que crear
tensor_trifocal.cpp en src/ e implementar los metodos marcados como override

se pueden agregar mas metodos, datos privados, clases extra, etc ya dependen de
lo que se necesite para implementar TrifocalTensor

Tambien hay que modificar src/main.cpp para que el objecto que repesenta la
simulacion nueva se genere al vuelo

class TrifocalTensor : public FormulationBase, public SimulatorInterface {

    public:
        TrifocalTensor(const boost::property_tree::ptree &parameters);
        ~TrifocalTensor() { };

        virtual Matrix_t GetH() override;
        virtual Matrix_t Getg() override;

        void Update(Matrix_t solution) override;
        void LogCurrentResults() override;
        bool Continue() override;


};

*/

#endif
