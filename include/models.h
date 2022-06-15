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

#ifndef MODELS_H
#define MODELS_H

#include <vector>
#include <memory>

#include <Eigen/Dense>
#include <cmath>
#include <boost/shared_ptr.hpp>
#include "common.h"


/******************************************************************************
 * Based on kajita's  dynamic model
 *  x_{k+1} = Ax_{k} + Bu
 *  z^{x} = Cx_{k}
 *
 * x is the state vector and u the control vector
 * C is a row vector
 *****************************************************************************/
class UnidimensionalIntegrator {
    protected:
        // system matrices
        Matrix_t A_, B_;

        // state vector
        Matrix_t state_vector_;

        void initialize_A_(const FPTYPE &T);
        void initialize_B_(const FPTYPE &T);
        void initialize_state_vector_(const FPTYPE &position, const FPTYPE &speed, const FPTYPE &acceleration);

    public:
        UnidimensionalIntegrator(const FPTYPE &T, const FPTYPE &position,  const FPTYPE &speed=0.0, const FPTYPE &acceleration=0.0);

        inline const Matrix_t &GetStateVector() const {
            return state_vector_;
        }
        inline const FPTYPE &GetPosition() const {
           return state_vector_(0, 0);
        }
        inline const FPTYPE &GetSpeed() const {
           return state_vector_(1, 0);
        }
        inline const FPTYPE &GetAcceleration() const {
           return state_vector_(2, 0);
        }
        inline virtual void UpdateState(const FPTYPE &jerk) {
            state_vector_ = A_*state_vector_ + B_*jerk;
        }
        inline FPTYPE GetNextPosition(const FPTYPE &jerk) const {
            return (A_*state_vector_ + B_*jerk)(0,0);
        }
        inline void SetPosition(const FPTYPE &position) {
            state_vector_(0) = position;
        }
        inline void SetVelocity(const FPTYPE &velocity) {
            state_vector_(1) = velocity;
        }
        inline void SetAcceleration(const FPTYPE &acceleration) {
            state_vector_(2) = acceleration;
        }
};

class FlyingFootUnidimensionalIntegrator : public UnidimensionalIntegrator {
    public:
        FlyingFootUnidimensionalIntegrator(const FPTYPE &T, const FPTYPE &position,  const FPTYPE &speed=0.0, const FPTYPE &acceleration=0.0);

        virtual inline void ResetState() {
            FPTYPE lastflying   = state_vector_(0, 0);
            state_vector_(0, 0) = support_foot_orientation_;
            state_vector_(1, 0) = 0.0;
            state_vector_(2, 0) = 0.0;
            // support_foot_orientation_ is the previous support foot orientation, in the previous CoM frame
            support_foot_orientation_ = lastflying;
        }
        inline const FPTYPE &GetSupport() const {
           return support_foot_orientation_;
        }
        inline void SetSupport(const FPTYPE &angle) {
            support_foot_orientation_ = angle;
        }
    protected:
        FPTYPE support_foot_orientation_;
};

class UnidimensionalCartModel : public UnidimensionalIntegrator {
    protected:
        // System matrices
        Matrix_t c_;

        // Output
        FPTYPE zmp_position_;

        void initialize_c_(const FPTYPE &CoM_height);

    public:
        UnidimensionalCartModel(const FPTYPE &T, const FPTYPE &CoM_height, const FPTYPE &zmp_position,
                                const FPTYPE &position,  const FPTYPE &speed=0.0, const FPTYPE &acceleration=0.0);

        inline const FPTYPE &GetZMPPosition() const {
            return zmp_position_;
        }
        void UpdateState(const FPTYPE &jerk) override;
};


/******************************************************************************
 *
 *****************************************************************************/

class RobotDynamicModel {

    public:
        RobotDynamicModel(const FPTYPE T, const FPTYPE CoM_height,
                          const FPTYPE zmp_x_position=0.0, const FPTYPE zmp_y_position=0.0,
                          const FPTYPE x_position=0.0,  const FPTYPE x_speed=0.0, const FPTYPE x_acceleration=0.0,
                          const FPTYPE y_position=0.0,  const FPTYPE y_speed=0.0, const FPTYPE y_acceleration=0.0,
                          const FPTYPE tcom_position=0.0,  const FPTYPE tcom_speed=0.0, const FPTYPE tcom_acceleration=0.0,
                          const FPTYPE tfoot_position=0.0,  const FPTYPE tfoot_speed=0.0, const FPTYPE tfoot_acceleration=0.0);

        inline const Matrix_t &GetXStateVector() const {
            return x_axis_model_.GetStateVector();
        }
        inline const FPTYPE &GetZMP_X_Position() const {
           return x_axis_model_.GetZMPPosition();
        }
        inline const FPTYPE &GetCoM_X_Position() const {
            return x_axis_model_.GetPosition();
        }
        inline const FPTYPE &GetCoM_X_Speed() const {
           return x_axis_model_.GetSpeed();
        }
        inline const FPTYPE &GetCoM_X_Acceleration() const {
            return x_axis_model_.GetAcceleration();
        }
        inline const Matrix_t &GetYStateVector() const {
            return y_axis_model_.GetStateVector();
        }
        inline const FPTYPE &GetZMP_Y_Position() const {
            return y_axis_model_.GetZMPPosition();
        }
        inline const FPTYPE &GetCoM_Y_Position() const {
            return y_axis_model_.GetPosition();
        }
        inline const FPTYPE &GetCoM_Y_Speed() const {
           return y_axis_model_.GetSpeed();
        }
        inline const FPTYPE &GetCoM_Y_Acceleration() const {
            return y_axis_model_.GetAcceleration();
        }
        inline const Matrix_t &GetTCOMStateVector() const {
            return tcom_model_.GetStateVector();
        }
        inline const Matrix_t &GetTFOOTStateVector() const {
            return tfoot_model_.GetStateVector();
        }
        inline const FPTYPE &GetTCOM_Position() const {
            return tcom_model_.GetPosition();
        }
        inline const FPTYPE GetTCOM_NextPosition(const FPTYPE &jerk) const {
            return tcom_model_.GetNextPosition(jerk);
        }
        inline const FPTYPE GetCOM_X_NextPosition(const FPTYPE &xjerk) const {
            return x_axis_model_.GetNextPosition(xjerk);
        }
        inline const FPTYPE GetCOM_Y_NextPosition(const FPTYPE &yjerk) const {
            return y_axis_model_.GetNextPosition(yjerk);
        }
        inline const FPTYPE &GetTFOOT_Position() const {
            return tfoot_model_.GetPosition();
        }
        inline const FPTYPE &GetTFOOTSupport() const {
            return tfoot_model_.GetSupport();
        }

        // Position-only
        void UpdateState(FPTYPE x_jerk, FPTYPE y_jerk);
        // Position and angles
        void UpdateState(FPTYPE x_jerk, FPTYPE y_jerk, FPTYPE tcom_jerk, FPTYPE tfoot_jerk);       
        void SetCoM_X_Position(FPTYPE position);
        void SetCoM_Y_Position(FPTYPE position);
        void SetCoM_X_Speed(FPTYPE speed);
        void SetCoM_Y_Speed(FPTYPE speed);
        void SetCoM_angle_Position(FPTYPE position);
        void SetFoot_angle_Position(FPTYPE position);
        // Position-only
        void ResetTranslationStateVector(FPTYPE angle);
        // Rotation-only
        void ResetOrientationStateVector();
        // A method to reset the flying foot state at each support change
        void ResetFlyingFootState();
    private:
        UnidimensionalCartModel x_axis_model_;
        UnidimensionalCartModel y_axis_model_;
        UnidimensionalIntegrator tcom_model_;
        FlyingFootUnidimensionalIntegrator tfoot_model_;        
};



/******************************************************************************
 *
 *****************************************************************************/

class MPCModel {

    public:
        MPCModel(const int N, const FPTYPE T, const FPTYPE CoM_height, unsigned short int s1=2, unsigned short int s2=9);

        inline const Matrix_t &get_Pps() const {
            return Pps_; 
        }
        
        inline const Matrix_t &get_Pvs() const {
            return Pvs_;
        }

        inline const Matrix_t &get_Pas() const {
            return Pas_;
        }

        inline const Matrix_t &get_Pzs() const {
            return Pzs_;
        }

        inline const Matrix_t &get_Ppu() const {
            return Ppu_;
        }

        inline const Matrix_t &get_Pvu() const {
            return Pvu_;
        }

        inline const Matrix_t &get_Pau() const {
            return Pau_;
        }

        inline const Matrix_t &get_Pzu() const {
            return Pzu_;
        }
        // For the flying foot
        inline const Matrix_t &get_Ppuk() const {
            return Ppuk_;
        }
        // For the flying foot
        inline const Matrix_t &get_ThetaV() const {
            return theta_V;
        }

        //For the flying foot
        void SetPpuk(unsigned short int s1, unsigned short int s2, const int N);
        void SetThetaV(unsigned short int s1,
                       unsigned short int s2,
                       const int N,
                       const FPTYPE support_foot_orientation_,
                       const Matrix_t &foot_state_);


    private:
        // Recursive matrices
        Matrix_t Pps_;
        Matrix_t Pvs_;
        Matrix_t Pas_;
        Matrix_t Pzs_;

        // For position
        Matrix_t Ppu_;
        // For velocity
        Matrix_t Pvu_;
        // For acceleration
        Matrix_t Pau_;
        // For the ZMP position
        Matrix_t Pzu_;
        // For the Flying foot
        Matrix_t Ppuk_;
        // For the flying foot
        Matrix_t theta_V;

        void InitializePps(const int N, const FPTYPE T);
        void InitializePvs(const int N, const FPTYPE T);
        void InitializePas(const int N, const FPTYPE T);
        void InitializePzs(const int N, const FPTYPE T, FPTYPE h);

        void InitializePpu(const int N, const FPTYPE T);
        void InitializePvu(const int N, const FPTYPE T);
        void InitializePau(const int N, const FPTYPE T);
        void InitializePzu(const int N, const FPTYPE T, FPTYPE h);
        // For the flying foot.
        void InitializePpuk(unsigned short int s1, unsigned short int s2, const int N);
        void InitializeThetaV(unsigned short int s1,
                              unsigned short int s2,
                              const int N,
                              const FPTYPE support_foot_orientation_);
};


/******************************************************************************
 *
 *                           Visual Features Models
 *
 *****************************************************************************/
class VisualConstraintEntryInterface {
    public:
        VisualConstraintEntryInterface(FPTYPE ck_x_t, FPTYPE ck_z_t, FPTYPE theta, FPTYPE phi) : ck_x_t_(ck_x_t), ck_z_t_(ck_z_t), theta_(theta), phi_(phi), gain_(0.0) {};
        virtual ~VisualConstraintEntryInterface() {};
        virtual std::pair<int, int> GetMatrixPosition() const = 0;
        virtual FPTYPE GetExpectedValue() const = 0;
        virtual std::string GetName() const = 0;  
        // setters 
        void phi(FPTYPE new_value)      { phi_  = new_value; };
        void gain(FPTYPE new_value)     { gain_ = new_value; };
        // getters
        const FPTYPE &GetGain()               { return gain_; };
        const FPTYPE &GetPhi()                { return phi_; };     
        virtual FPTYPE a() = 0;
        virtual FPTYPE b() = 0;
        virtual FPTYPE c() = 0;

        virtual FPTYPE predicted_value(const FPTYPE Xk1,
                                       const FPTYPE Yk1,
                                       const FPTYPE theta) const = 0 ;
         // setters 
        void ck_x_t(FPTYPE new_value)   { ck_x_t_ = new_value; };
        void ck_z_t(FPTYPE new_value)   { ck_z_t_ = new_value; };
        void theta(FPTYPE new_value)    { theta_  = new_value; };
        void c_x_com(FPTYPE new_value)  { c_x_com_= new_value; };
        void c_z_com(FPTYPE new_value)  { c_z_com_= new_value; };
    protected:
        FPTYPE phi_;
        FPTYPE gain_;
        FPTYPE ck_x_t_;
        FPTYPE ck_z_t_;
        FPTYPE theta_;
        //Displacement between the camera and CoM
        FPTYPE c_x_com_;
        FPTYPE c_z_com_;
};


class HomographyEntryInterface : public VisualConstraintEntryInterface {
    public:
        HomographyEntryInterface(FPTYPE d, FPTYPE nz, FPTYPE nx, FPTYPE ck_x_t, FPTYPE ck_z_t, FPTYPE theta, FPTYPE phi) :
             VisualConstraintEntryInterface(ck_x_t,ck_z_t,theta,phi), d_(d), nz_(nz), nx_(nx) {};
        virtual ~HomographyEntryInterface() {};
        virtual std::string GetName() const override = 0;
        // setters 
        void d(FPTYPE new_value)        { d_  = new_value; };
        void nz(FPTYPE new_value)       { nz_ = new_value; };
        void nx(FPTYPE new_value)       { nx_ = new_value; };
        void ny(FPTYPE new_value)       { ny_ = new_value; };

        // The following methods are used in the non-linear case
        virtual Matrix_t GetHX(const Matrix_t &thetas) const = 0 ;
        virtual Matrix_t GetHY(const Matrix_t &thetas) const = 0 ;
        virtual Matrix_t GetHT(const Matrix_t &Xk1,
                               const Matrix_t &Yk1,
                               const Matrix_t &thetas) const = 0 ;
        virtual Matrix_t GetRE(const Matrix_t &Xk1,
                               const Matrix_t &Yk1,
                               const Matrix_t &thetas) const = 0 ;
    protected:
        FPTYPE d_;
        FPTYPE nz_;
        FPTYPE nx_;
        FPTYPE ny_;
};

class EssentialEntryInterface  : public VisualConstraintEntryInterface {
    public:
        EssentialEntryInterface(FPTYPE ck_x_t_,FPTYPE ck_x_t, FPTYPE ck_z_t, FPTYPE theta, FPTYPE phi) :
            VisualConstraintEntryInterface(ck_x_t,ck_z_t,theta,phi) {};
        virtual ~EssentialEntryInterface() { };
        virtual std::string GetName() const = 0 ;

        // Setters
        inline void ck_y_t(FPTYPE new_value)   { ck_y_t_ = new_value; };
        inline void y_real(FPTYPE new_value)   { y_real_ = new_value; };

        // The following methods are used in the non-linear case
        virtual Matrix_t GetHX(const Matrix_t &phis) const = 0 ;
        virtual Matrix_t GetHY(const Matrix_t &phis) const = 0 ;
        virtual Matrix_t GetHT(const Matrix_t &Xk1,
                               const Matrix_t &Yk1,
                               const Matrix_t &phis) const = 0 ;
        virtual Matrix_t GetRE(const Matrix_t &Xk1,
                               const Matrix_t &Yk1,
                               const Matrix_t &phis) const = 0 ;

    protected:
        FPTYPE ck_y_t_;
        FPTYPE y_real_;
};

class HomographyH13 : public HomographyEntryInterface {
    public:
        HomographyH13() : HomographyEntryInterface(0, 0, 0, 0, 0, 0, 0) {};
        ~HomographyH13() { };
        inline std::pair<int, int> GetMatrixPosition() const override {return std::make_pair(0, 2);};
        inline FPTYPE GetExpectedValue() const override { return (FPTYPE) 0.0;};
        inline std::string GetName() const override {return "h13";};
        inline FPTYPE a() override {return -std::sin(phi_) * (-nx_*std::sin(theta_)+nz_*std::cos(theta_)) / d_;};
        inline FPTYPE b() override {return -std::cos(phi_) * (-nx_*std::sin(theta_)+nz_*std::cos(theta_)) / d_;};
        inline FPTYPE c() override {return ((-nx_*std::sin(theta_)+nz_*std::cos(theta_)) * (-ck_x_t_*d_ * std::cos(phi_) + ck_z_t_ *d_* std::sin(phi_) + c_x_com_*(std::cos(phi_)-std::cos(phi_+theta_)) + c_z_com_*(std::sin(phi_+theta_)-std::sin(phi_)) ) / d_ ) - std::sin(phi_ + theta_);};

        inline FPTYPE predicted_value(const FPTYPE Xk1,
                                      const FPTYPE Yk1,
                                      const FPTYPE theta) const override
        {
            return ((-nx_*std::sin(theta)+nz_*std::cos(theta))*(-Xk1+ck_z_t_*d_))*std::sin(phi_) / d_ + ((-nx_*std::sin(theta)+nz_*std::cos(theta))*(-Yk1-ck_x_t_*d_))*std::cos(phi_) / d_ + ((-nx_*std::sin(theta)+nz_*std::cos(theta))*(std::cos(phi_)-std::cos(phi_+theta)))*c_x_com_ / d_ + ((-nx_*std::sin(theta)+nz_*std::cos(theta))*(std::sin(phi_+theta)-std::sin(phi_)))*c_z_com_ / d_ - std::sin(theta+phi_);
        }

        //Nonlinear case
        inline Matrix_t GetHX(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = (-(-nx_*std::sin(thetas(i, 0))+nz_*std::cos(thetas(i, 0)))*std::sin(phi_) / d_);
            }
            return H;
        }

        inline Matrix_t GetHY(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = (-(-nx_*std::sin(thetas(i, 0))+nz_*std::cos(thetas(i, 0)))*std::cos(phi_) / d_);
            }

            return H;
        }

        //GradSIN matlab's code
        inline Matrix_t GetHT(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override{
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = -(nx_*std::cos(thetas(i, 0))+nz_*std::sin(thetas(i, 0))) * ( (-Xk1(i,0)+ck_z_t_*d_)*std::sin(phi_) - (ck_x_t_*d_+Yk1(i,0))*std::cos(phi_) ) / d_ - std::cos(phi_ + thetas(i, 0)) ;
            }
            return H;
        }

        inline Matrix_t GetRE(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override {
            Matrix_t re(thetas.rows(),1);
            for (int i = 0; i < thetas.rows(); i++) {
                re(i, 0) = ((-nx_*std::sin(thetas(i, 0))+nz_*std::cos(thetas(i, 0)))*(-Xk1(i,0)+ck_z_t_*d_))*std::sin(phi_) / d_ - ((-nx_*std::sin(thetas(i, 0))+nz_*std::cos(thetas(i, 0)))*(ck_x_t_*d_+Yk1(i,0)))*std::cos(phi_) / d_ - std::sin(thetas(i,0)+phi_);
            }
            return re;
        }
};

class HomographyH33 : public HomographyEntryInterface {
    public:
        HomographyH33() :  HomographyEntryInterface(0, 0, 0, 0, 0, 0, 0) {};
        ~HomographyH33() {};
        inline std::pair<int, int> GetMatrixPosition() const override {return std::make_pair(2, 2);};
        inline FPTYPE GetExpectedValue() const override { return (FPTYPE) 1.0;};
        inline std::string GetName() const override {return "h33";};
        inline FPTYPE a() override {return   (-nx_*std::sin(theta_)+nz_*std::cos(theta_)) * std::cos(phi_) / d_;};
        inline FPTYPE b() override {return  -(-nx_*std::sin(theta_)+nz_*std::cos(theta_)) * std::sin(phi_) / d_;};
        inline FPTYPE c() override {return (-(-nx_*std::sin(theta_)+nz_*std::cos(theta_)) * (ck_x_t_*d_* std::sin(phi_) + ck_z_t_*d_* std::cos(phi_) + c_x_com_*(std::sin(phi_)-std::sin(phi_+theta_)) + c_z_com_*(std::cos(phi_)-std::cos(phi_+theta_))   ) /d_ ) + std::cos(phi_ + theta_);};

        inline FPTYPE predicted_value(const FPTYPE Xk1,
                                      const FPTYPE Yk1,
                                      const FPTYPE theta) const override
        {
            return ((-nx_*std::sin(theta)+nz_*std::cos(theta))*(Xk1-ck_z_t_*d_))*std::cos(phi_) / d_ - ((-nx_*std::sin(theta)+nz_*std::cos(theta))*(ck_x_t_*d_+Yk1))*std::sin(phi_) / d_ + ((-nx_*std::sin(theta)+nz_*std::cos(theta))*(std::sin(phi_)-std::sin(phi_+theta)))*c_x_com_ / d_ + ((-nx_*std::sin(theta)+nz_*std::cos(theta))*(std::cos(phi_)-std::cos(phi_+theta)))*c_z_com_ / d_ + std::cos(theta+phi_);
        }

        //Nonlinear case
        inline Matrix_t GetHX(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = ((-nx_*std::sin(thetas(i, 0))+nz_*std::cos(thetas(i, 0)))*std::cos(phi_) / d_);
            }

            return H;
        }

        inline Matrix_t GetHY(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = (-(-nx_*std::sin(thetas(i, 0))+nz_*std::cos(thetas(i, 0)))*std::sin(phi_) / d_);
            }

            return H;
        }

        //GradSIN matlab's code
        inline Matrix_t GetHT(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override{
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = -(nx_*std::cos(thetas(i, 0))+nz_*std::sin(thetas(i, 0))) * ( (Xk1(i,0)-ck_z_t_*d_)*std::cos(phi_) - (ck_x_t_*d_+Yk1(i,0))*std::sin(phi_) ) / d_ - std::sin(phi_ + thetas(i, 0)) ;
            }
            return H;
        }

        inline Matrix_t GetRE(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override {
            Matrix_t re(thetas.rows(),1);
            for (int i = 0; i < thetas.rows(); i++) {
                re(i, 0) = ((-nx_*std::sin(thetas(i, 0))+nz_*std::cos(thetas(i, 0)))*(Xk1(i,0)-ck_z_t_*d_))*std::cos(phi_) / d_ - ((-nx_*std::sin(thetas(i, 0))+nz_*std::cos(thetas(i, 0)))*(ck_x_t_*d_+Yk1(i,0)))*std::sin(phi_) / d_ + std::cos(thetas(i,0)+phi_);
            }
            return re;
        }
};

class HomographyH31 : public HomographyEntryInterface {
    public:
        HomographyH31() : HomographyEntryInterface(0, 0, 0, 0, 0, 0, 0) {};
        ~HomographyH31() {};
        inline std::pair<int, int> GetMatrixPosition() const override {return std::make_pair(2, 0);};
        inline FPTYPE GetExpectedValue() const override { return (FPTYPE) 0.0;};
        inline std::string GetName() const override {return "h31";};
        inline FPTYPE a() override {return  (nx_*std::cos(theta_)+nz_*std::sin(theta_)) * std::cos(phi_) / d_;};
        inline FPTYPE b() override {return -(nx_*std::cos(theta_)+nz_*std::sin(theta_)) * std::sin(phi_) / d_;};
        inline FPTYPE c() override {return (-(nx_*std::cos(theta_)+nz_*std::sin(theta_)) * (ck_x_t_*d_*std::sin(phi_) + ck_z_t_*d_*std::cos(phi_) + c_x_com_*(std::sin(phi_)-std::sin(phi_+theta_)) + c_z_com_*(std::cos(phi_)-std::cos(phi_+theta_)) ) /d_) + std::sin(phi_ + theta_);};

        inline FPTYPE predicted_value(const FPTYPE Xk1,
                                      const FPTYPE Yk1,
                                      const FPTYPE theta) const override
        {
            return ((nx_*std::cos(theta)+nz_*std::sin(theta))*(Xk1-ck_z_t_*d_))*std::cos(phi_) / d_ - ((nx_*std::cos(theta)+nz_*std::sin(theta)) *(ck_x_t_*d_+Yk1))*std::sin(phi_) / d_ + ((nx_*std::cos(theta)+nz_*std::sin(theta))*(std::sin(phi_)-std::sin(phi_+theta)))*c_x_com_/ d_ + ((nx_*std::cos(theta)+nz_*std::sin(theta))*(std::cos(phi_)-std::cos(phi_+theta)))*c_z_com_/ d_ + std::sin(theta+phi_);
        }

        //Nonlinear case
        inline Matrix_t GetHX(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = ((nx_*std::cos(thetas(i, 0))+nz_*std::sin(thetas(i, 0)))*std::cos(phi_) / d_);
            }

            return H;
        }

        inline Matrix_t GetHY(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = (-(nx_*std::cos(thetas(i, 0))+nz_*std::sin(thetas(i, 0)))*std::sin(phi_) / d_);
            }

            return H;
        }

        //GradSIN matlab's code
        inline Matrix_t GetHT(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override{
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = (-nx_*std::sin(thetas(i, 0))+nz_*std::cos(thetas(i, 0))) * ( (Xk1(i,0)-ck_z_t_*d_)*std::cos(phi_) - (ck_x_t_*d_+Yk1(i,0))*std::sin(phi_) ) / d_ + std::cos(phi_ + thetas(i, 0)) ;
            }
            return H;
        }

        inline Matrix_t GetRE(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override {
            Matrix_t re(thetas.rows(),1);
            for (int i = 0; i < thetas.rows(); i++) {
                re(i, 0) = ((nx_*std::cos(thetas(i, 0))+nz_*std::sin(thetas(i, 0)))*(Xk1(i, 0)-ck_z_t_*d_))*std::cos(phi_) / d_ - ((nx_*std::cos(thetas(i, 0))+nz_*std::sin(thetas(i, 0)))*(ck_x_t_*d_+Yk1(i, 0)))*std::sin(phi_) / d_ + std::sin(thetas(i,0)+phi_);
            }
            return re;
        }
};

class HomographyH11 : public HomographyEntryInterface {
    public:
        HomographyH11() : HomographyEntryInterface(0, 0, 0, 0, 0, 0, 0) {};
        ~HomographyH11() {};
        inline std::pair<int, int> GetMatrixPosition() const override {return std::make_pair(0, 0);};
        inline FPTYPE GetExpectedValue() const override { return (FPTYPE) 1.0;};
        inline std::string GetName() const override {return "h11";};
        inline FPTYPE a() override {return -(nx_*std::cos(theta_)+nz_*std::sin(theta_)) * std::sin(phi_) / d_;};
        inline FPTYPE b() override {return -(nx_*std::cos(theta_)+nz_*std::sin(theta_)) * std::cos(phi_) / d_;};
        inline FPTYPE c() override {return ((nx_*std::cos(theta_)+nz_*std::sin(theta_)) * (-ck_x_t_*d_*std::cos(phi_) + ck_z_t_*d_*std::sin(phi_) + c_x_com_*(std::cos(phi_)-std::cos(phi_+theta_)) + c_z_com_*(std::sin(phi_+theta_)-std::sin(phi_))  ) /d_ ) + std::cos(phi_ + theta_);};
        inline FPTYPE predicted_value(const FPTYPE Xk1,
                                      const FPTYPE Yk1,
                                      const FPTYPE theta) const override
        {
            return ((nx_*std::cos(theta)+nz_*std::sin(theta))*(-Xk1+ck_z_t_*d_))*std::sin(phi_) / d_ - ((nx_*std::cos(theta)+nz_*std::sin(theta))*(ck_x_t_*d_+Yk1))*std::cos(phi_) / d_ + ((nx_*std::cos(theta)+nz_*std::sin(theta))*(std::cos(phi_)-std::cos(phi_+theta)))*c_x_com_/ d_ + ((nx_*std::cos(theta)+nz_*std::sin(theta))*(std::sin(phi_+theta)-std::sin(phi_)))*c_z_com_/ d_ + std::cos(theta+phi_);
        }

        //Nonlinear case
        inline Matrix_t GetHX(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = (-(nx_*std::cos(thetas(i, 0))+nz_*std::sin(thetas(i, 0)))*std::sin(phi_) / d_);
            }

            return H;
        }

        inline Matrix_t GetHY(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = (-(nx_*std::cos(thetas(i, 0))+nz_*std::sin(thetas(i, 0)))*std::cos(phi_) / d_);
            }

            return H;
        }

        //GradSIN matlab's code
        inline Matrix_t GetHT(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override{
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = (-nx_*std::sin(thetas(i, 0))+nz_*std::cos(thetas(i, 0))) * ( (-Xk1(i,0)+ck_z_t_*d_)*std::sin(phi_) - (ck_x_t_*d_+Yk1(i,0))*std::cos(phi_) ) / d_ - std::sin(phi_ + thetas(i, 0)) ;
            }
            return H;
        }

        inline Matrix_t GetRE(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override {
            Matrix_t re(thetas.rows(),1);
            for (int i = 0; i < thetas.rows(); i++) {
                re(i, 0) = ((nx_*std::cos(thetas(i, 0))+nz_*std::sin(thetas(i, 0)))*(-Xk1(i,0)+ck_z_t_*d_))*std::sin(phi_) / d_ - ((nx_*std::cos(thetas(i, 0))+nz_*std::sin(thetas(i, 0)))*(ck_x_t_*d_+Yk1(i,0)))*std::cos(phi_) / d_ + std::cos(thetas(i,0)+phi_);
            }
            return re;
        }
};

class HomographyH12 : public HomographyEntryInterface {
public:
    HomographyH12() : HomographyEntryInterface(0, 0, 0, 0, 0, 0, 0) {};
    ~HomographyH12() {};
    inline std::pair<int, int> GetMatrixPosition() const override {return std::make_pair(0, 1);};
    inline FPTYPE GetExpectedValue() const override { return (FPTYPE) 0.0;};
    inline std::string GetName() const override {return "h12";};
    inline FPTYPE a() override {return -ny_ * std::sin(phi_) / d_;};
    inline FPTYPE b() override {return -ny_ * std::cos(phi_) / d_;};
    inline FPTYPE c() override {return (ny_ * (-ck_x_t_*d_*std::cos(phi_) + ck_z_t_*d_*std::sin(phi_) + c_x_com_*(std::cos(phi_)-std::cos(phi_+theta_)) + c_z_com_*(std::sin(phi_+theta_)-std::sin(phi_))  )/d_ ) ;};
    inline FPTYPE predicted_value(const FPTYPE Xk1,
                                  const FPTYPE Yk1,
                                  const FPTYPE theta) const override
    {
        return (ny_*(-Xk1+ck_z_t_*d_))*std::sin(phi_) / d_ + (ny_*(-ck_x_t_*d_-Yk1))*std::cos(phi_) / d_ + ny_*(std::cos(phi_)-std::cos(phi_+theta))*c_x_com_/ d_ + ny_*(std::sin(phi_+theta)-std::sin(phi_))*c_z_com_/ d_;
    }

    //Nonlinear case
    inline Matrix_t GetHX(const Matrix_t &thetas) const override {
        Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
        H = (-ny_*std::sin(phi_) / d_) * Matrix_t::Identity(thetas.rows(),thetas.rows());
        return H;
    }

    inline Matrix_t GetHY(const Matrix_t &thetas) const override {
        Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
        H = (-ny_*std::cos(phi_) / d_) * Matrix_t::Identity(thetas.rows(),thetas.rows());
        return H;
    }

    //GradCOS matlab's code
    inline Matrix_t GetHT(const Matrix_t &Xk1,
                          const Matrix_t &Yk1,
                          const Matrix_t &thetas) const override{
        Matrix_t H(thetas.rows(),thetas.rows());
        H.setZero();

        return H;
    }

    inline Matrix_t GetRE(const Matrix_t &Xk1,
                          const Matrix_t &Yk1,
                          const Matrix_t &thetas) const override {
        Matrix_t re(thetas.rows(),1);
        for (int i = 0; i < thetas.rows(); i++) {
            re(i, 0) = (ny_*(-Xk1(i,0)+ck_z_t_*d_))*std::sin(phi_) / d_ + (ny_*(-ck_x_t_*d_-Yk1(i,0)))*std::cos(phi_) / d_ ;
        }
        return re;
    }
};

class HomographyH32 : public HomographyEntryInterface {
public:
    HomographyH32() : HomographyEntryInterface(0, 0, 0, 0, 0, 0, 0) {};
    ~HomographyH32() {};
    inline std::pair<int, int> GetMatrixPosition() const override {return std::make_pair(2, 1);};
    inline FPTYPE GetExpectedValue() const override { return (FPTYPE) 0.0;};
    inline std::string GetName() const override {return "h32";};
    inline FPTYPE a() override {return  ny_ * std::cos(phi_) / d_;};
    inline FPTYPE b() override {return -ny_ * std::sin(phi_) / d_;};
    inline FPTYPE c() override {return (ny_ * (-ck_x_t_*d_*std::sin(phi_) - ck_z_t_*d_*std::cos(phi_) + c_x_com_*(std::sin(phi_)-std::sin(phi_+theta_)) + c_z_com_*(std::cos(phi_)-std::cos(phi_+theta_)) )/d_ ) ;};
    inline FPTYPE predicted_value(const FPTYPE Xk1,
                                  const FPTYPE Yk1,
                                  const FPTYPE theta) const override
    {
        return (ny_*(Xk1-ck_z_t_*d_))*std::cos(phi_) / d_ - (ny_*(ck_x_t_*d_+Yk1))*std::sin(phi_) / d_ + (ny_*(std::sin(phi_)-std::sin(phi_+theta)))*c_x_com_/ d_ + (ny_*(std::cos(phi_)-std::cos(phi_+theta)))*c_z_com_/ d_ ;
    }

    //Nonlinear case
    inline Matrix_t GetHX(const Matrix_t &thetas) const override {
        Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
        H = (ny_*std::cos(phi_) / d_) * Matrix_t::Identity(thetas.rows(),thetas.rows());
        return H;
    }

    inline Matrix_t GetHY(const Matrix_t &thetas) const override {
        Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
        H = (-ny_*std::sin(phi_) / d_) * Matrix_t::Identity(thetas.rows(),thetas.rows());
        return H;
    }

    //GradCOS matlab's code
    inline Matrix_t GetHT(const Matrix_t &Xk1,
                          const Matrix_t &Yk1,
                          const Matrix_t &thetas) const override{
        Matrix_t H(thetas.rows(),thetas.rows());
        H.setZero();

        return H;
    }

    inline Matrix_t GetRE(const Matrix_t &Xk1,
                          const Matrix_t &Yk1,
                          const Matrix_t &thetas) const override {
        Matrix_t re(thetas.rows(),1);
        for (int i = 0; i < thetas.rows(); i++) {
            re(i, 0) = (ny_*(Xk1(i,0)-ck_z_t_*d_))*std::cos(phi_) / d_ - (ny_*(ck_x_t_*d_+Yk1(i,0)))*std::sin(phi_) / d_ ;
        }
        return re;
    }
};

class EssentialE12 : public EssentialEntryInterface {
    public:
        EssentialE12() : EssentialEntryInterface(0, 0, 0, 0, 0){};
        ~EssentialE12() {};
        inline std::pair<int, int> GetMatrixPosition() const override {return std::make_pair(0, 1);};
        inline FPTYPE GetExpectedValue() const override { return (FPTYPE) 0.0;};
        inline std::string GetName() const override {return "e12";};
        inline FPTYPE a() override {return std::cos(theta_)/(y_real_*std::cos(phi_ + theta_));};
        inline FPTYPE b() override {return std::sin(theta_)/(y_real_*std::cos(phi_ + theta_));};
        inline FPTYPE c() override {return ((ck_x_t_*y_real_/ck_y_t_)*std::sin(theta_) - (ck_z_t_*y_real_/ck_y_t_)*std::cos(theta_))/(y_real_*std::cos(phi_ + theta_));};
        inline Matrix_t GetHX(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = std::cos(thetas(i, 0))/(y_real_*std::cos(phi_ + thetas(i, 0)));
            }
            return H;
        }
        inline Matrix_t GetHY(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = std::sin(thetas(i, 0))/(y_real_*std::cos(phi_ + thetas(i, 0)));
            }
            return H;
        }
        inline Matrix_t GetHT(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = ((Yk1(i,0)+(ck_x_t_*y_real_/ck_y_t_))*std::cos(phi_)+(Xk1(i,0)-(ck_z_t_*y_real_/ck_y_t_))*std::sin(phi_))/(y_real_*std::cos(phi_ + thetas(i, 0))*std::cos(phi_ + thetas(i, 0)));
            }
            return H;
        }
        inline Matrix_t GetRE(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override {
            Matrix_t re(thetas.rows(),1);
            for (int i = 0; i < thetas.rows(); i++) {
                re(i, 0) = ((Xk1(i,0)-(ck_z_t_*y_real_/ck_y_t_))*std::cos(thetas(i, 0)) + ((ck_x_t_*y_real_/ck_y_t_)+Yk1(i,0))*std::sin(thetas(i, 0)))/(y_real_*std::cos(phi_ + thetas(i, 0)));
            }
            return re;
        }
        inline FPTYPE predicted_value(const FPTYPE Xk1,
                                      const FPTYPE Yk1,
                                      const FPTYPE theta) const override {
            return ((Xk1-(ck_z_t_*y_real_/ck_y_t_))*std::cos(theta)+((ck_x_t_*y_real_/ck_y_t_)+Yk1)*std::sin(theta))/(y_real_*std::cos(phi_ + theta_));
        }
};

class EssentialE21 : public EssentialEntryInterface {
    public:
        EssentialE21() : EssentialEntryInterface(0, 0, 0, 0, 0){};
        ~EssentialE21() {};
        inline std::pair<int, int> GetMatrixPosition() const override {return std::make_pair(1, 0);};
        inline FPTYPE GetExpectedValue() const override { return (FPTYPE) 0.0;};
        inline std::string GetName() const override {return "e21";};
        inline FPTYPE a() override {return -std::cos(phi_)/(y_real_*std::cos(phi_ + theta_));};
        inline FPTYPE b() override {return  std::sin(phi_)/(y_real_*std::cos(phi_ + theta_));};
        inline FPTYPE c() override {return  ((ck_x_t_*y_real_/ck_y_t_)*std::sin(phi_) + (ck_z_t_*y_real_/ck_y_t_)*std::cos(phi_))/(y_real_*std::cos(phi_ + theta_));};
        inline Matrix_t GetHX(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = -std::cos(phi_)/(y_real_*std::cos(phi_ + thetas(i, 0)));
            }
            return H;        
        }
        inline Matrix_t GetHY(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) =  std::sin(phi_)/(y_real_*std::cos(phi_ + thetas(i, 0)));
            }
            return H;        
        }
        inline Matrix_t GetHT(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = ((Yk1(i,0)+(ck_x_t_*y_real_/ck_y_t_))*std::sin(phi_)*std::sin(phi_ + thetas(i, 0)) +(-Xk1(i,0)+(ck_z_t_*y_real_/ck_y_t_))*std::sin(phi_ + thetas(i, 0))*std::cos(phi_))/(y_real_*std::cos(phi_ + thetas(i, 0))*std::cos(phi_ + thetas(i, 0)));
            }
            return H;
        }
        inline Matrix_t GetRE(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override {
            Matrix_t re(thetas.rows(),1);
            for (int i = 0; i < thetas.rows(); i++) {
                re(i, 0) = ((-Xk1(i,0)+(ck_z_t_*y_real_/ck_y_t_))*std::cos(phi_)+(Yk1(i,0)+(ck_x_t_*y_real_/ck_y_t_))*std::sin(phi_))/(y_real_*std::cos(phi_ + thetas(i, 0)));
            }
            return re;
        }
        inline FPTYPE predicted_value(const FPTYPE Xk1,
                                      const FPTYPE Yk1,
                                      const FPTYPE theta) const override {
            return ((-Xk1+(ck_z_t_*y_real_/ck_y_t_))*std::cos(phi_)+(Yk1+(ck_x_t_*y_real_/ck_y_t_))*std::sin(phi_))/(y_real_*std::cos(phi_ + theta_));
        }
};

class EssentialE23 : public EssentialEntryInterface {
    public:
        EssentialE23() : EssentialEntryInterface(0, 0, 0, 0, 0){};
        ~EssentialE23() {};
        inline std::pair<int, int> GetMatrixPosition() const override {return std::make_pair(1, 2);};
        inline FPTYPE GetExpectedValue() const override { return (FPTYPE) 0.0;};
        inline std::string GetName() const override {return "e23";};
        inline FPTYPE a() override {return  -std::sin(phi_)/(y_real_*std::cos(phi_ + theta_));};
        inline FPTYPE b() override {return  -std::cos(phi_)/(y_real_*std::cos(phi_ + theta_));};
        inline FPTYPE c() override {return  (-(ck_x_t_*y_real_/ck_y_t_)*std::cos(phi_) + (ck_z_t_*y_real_/ck_y_t_)*std::sin(phi_))/(y_real_*std::cos(phi_ + theta_));};
        inline Matrix_t GetHX(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = -std::sin(phi_)/(y_real_*std::cos(phi_ + thetas(i, 0)));
            }
            return H;        
        }
        inline Matrix_t GetHY(const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = -std::cos(phi_)/(y_real_*std::cos(phi_ + thetas(i, 0)));
            }
            return H;        
        }
        inline Matrix_t GetHT(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override {
            Matrix_t H(thetas.rows(),thetas.rows()); H.setZero();
            for (int i = 0; i < thetas.rows(); i++) {
                H(i, i) = ((-Yk1(i,0)-(ck_x_t_*y_real_/ck_y_t_))*std::cos(phi_)*std::sin(phi_ + thetas(i, 0)) +(-Xk1(i,0)+(ck_z_t_*y_real_/ck_y_t_))*std::sin(phi_ + thetas(i, 0))*std::sin(phi_))/(y_real_*std::cos(phi_ + thetas(i, 0))*std::cos(phi_ + thetas(i, 0)));
            }
            return H;
        }
        inline Matrix_t GetRE(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &thetas) const override {
            Matrix_t re(thetas.rows(),1);
            for (int i = 0; i < thetas.rows(); i++) {
                re(i, 0) = (-(+Yk1(i,0)+(ck_x_t_*y_real_/ck_y_t_))*std::cos(phi_)+(-Xk1(i,0)+(ck_z_t_*y_real_/ck_y_t_))*std::sin(phi_))/(y_real_*std::cos(phi_ + thetas(i, 0)));
            }
            return re;
        }
        inline FPTYPE predicted_value(const FPTYPE Xk1,
                                      const FPTYPE Yk1,
                                      const FPTYPE theta) const override {
            return ((-Yk1-(ck_x_t_*y_real_/ck_y_t_))*std::cos(phi_)+(-Xk1+(ck_z_t_*y_real_/ck_y_t_))*std::sin(phi_))/(y_real_*std::cos(phi_ + theta_));
        }
};

class EssentialE32 : public EssentialEntryInterface {
    public:
        EssentialE32() : EssentialEntryInterface(0, 0, 0, 0, 0){};
        ~EssentialE32() {};
        inline std::pair<int, int> GetMatrixPosition() const override {return std::make_pair(2, 1);};
        inline FPTYPE GetExpectedValue() const override { return (FPTYPE) 0.0;};
        inline std::string GetName() const override {return "e32";};
        inline FPTYPE a() override {return  -std::sin(theta_)/(y_real_*std::cos(phi_ + theta_));};
        inline FPTYPE b() override {return  std::cos(theta_)/(y_real_*std::cos(phi_ + theta_));};
        inline FPTYPE c() override {return  ((ck_x_t_*y_real_/ck_y_t_)*std::cos(theta_) + (ck_z_t_*y_real_/ck_y_t_)*std::sin(theta_))/(y_real_*std::cos(phi_ + theta_));};
        inline Matrix_t GetHX(const Matrix_t &phis) const override {
            Matrix_t Hessian(phis.rows(),phis.rows()); Hessian.setZero();
            for (int i = 0; i < phis.rows(); i++) {
                Hessian(i, i) = -std::sin(phis(i, 0))/(y_real_*std::cos(phi_ + phis(i, 0)));
            }
            return Hessian;
        }
        inline Matrix_t GetHY(const Matrix_t &phis) const override {
            Matrix_t Hessian(phis.rows(),phis.rows()); Hessian.setZero();
            for (int i = 0; i < phis.rows(); i++) {
                Hessian(i, i) = +std::cos(phis(i, 0))/(y_real_*std::cos(phi_ + phis(i, 0)));
            }
            return Hessian;
        }
        inline Matrix_t GetHT(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &phis) const override {
            Matrix_t Hessian(phis.rows(),phis.rows()); Hessian.setZero();
            for (int i = 0; i < phis.rows(); i++) {
                Hessian(i, i) = ((-Xk1(i,0)+(ck_z_t_*y_real_/ck_y_t_))*std::cos(phi_) + ((ck_x_t_*y_real_/ck_y_t_)+Yk1(i,0))*std::sin(phi_))/(y_real_*std::cos(phi_ + phis(i, 0))*std::cos(phi_ + phis(i, 0)));
            }
            return Hessian;
        }
        inline Matrix_t GetRE(const Matrix_t &Xk1,
                              const Matrix_t &Yk1,
                              const Matrix_t &phis) const override {
            Matrix_t re(phis.rows(),1);
            for (int i = 0; i < phis.rows(); i++) {
                re(i, 0) = ((Yk1(i,0)+(ck_x_t_*y_real_/ck_y_t_))*std::cos(phis(i, 0))+(-Xk1(i,0)+(ck_z_t_*y_real_/ck_y_t_))*std::sin(phis(i, 0)))/(y_real_*std::cos(phi_ + phis(i, 0)));
            }
            return re;
        }
        inline FPTYPE predicted_value(const FPTYPE Xk1,
                                      const FPTYPE Yk1,
                                      const FPTYPE theta) const override {
            return ((Yk1+(ck_x_t_*y_real_/ck_y_t_))*std::cos(theta)+(-Xk1+(ck_z_t_*y_real_/ck_y_t_))*std::sin(theta))/(y_real_*std::cos(phi_ + theta_));
        }
};

/******************************************************************************
 *
 *                           MPC Visual Features Models
 *
 *****************************************************************************/
class MPCVisualConstraint {
    public:
        MPCVisualConstraint(std::string type, int N) : N_(N) {};
        virtual ~MPCVisualConstraint() {}; 
        virtual FPTYPE  Predicted(const FPTYPE Xk1,
                                  const FPTYPE Yk1,
                                  const FPTYPE theta) const {
            return vconstraint_entry_hxx_->predicted_value(Xk1,Yk1,theta);
        };
        std::pair<int, int> GetMatrixPosition() const {
            return vconstraint_entry_hxx_->GetMatrixPosition();
        }
        inline int GetExpectedValue() const {
            return vconstraint_entry_hxx_->GetExpectedValue();
        }
        inline std::string GetName() const {
            return vconstraint_entry_hxx_->GetName();
        }
        inline const FPTYPE &GetGain() const {
            return vconstraint_entry_hxx_->GetGain();
        }
        inline void gain(const FPTYPE &new_value) {
            vconstraint_entry_hxx_->gain(new_value);
        }
        inline void phi(const FPTYPE &new_value) {
            vconstraint_entry_hxx_->phi(new_value);
        }
        inline void ck_x_t(const FPTYPE &new_value) {
            vconstraint_entry_hxx_->ck_x_t(new_value);
        }
        inline void ck_z_t(const FPTYPE &new_value) {
            vconstraint_entry_hxx_->ck_z_t(new_value);
        }
        inline void c_x_com(const FPTYPE &new_value){
            vconstraint_entry_hxx_->c_x_com(new_value);
        }
        inline void c_z_com(const FPTYPE &new_value){
            vconstraint_entry_hxx_->c_z_com(new_value);
        }
    protected:
        int N_;
        boost::shared_ptr<VisualConstraintEntryInterface> vconstraint_entry_hxx_;
};

class MPCLinearVisualConstraint : public MPCVisualConstraint {
    public:
        MPCLinearVisualConstraint(std::string type, int N) : MPCVisualConstraint(type,N), thetas_(N,1) {
            thetas_.setZero();
        };
        ~MPCLinearVisualConstraint() {};         
        virtual Matrix_t A() const = 0;
        virtual Matrix_t B() const = 0;
        virtual Matrix_t C() const = 0;
        inline void SetHp(const Matrix_t &Hp){
            if (Hp.rows() != N_) {
                throw std::domain_error("ERROR: thetas size should be equal to N (prediction horizon)");
            }
            Hp_ = Hp;
        }
        inline Matrix_t GetHp() { return Hp_; };
        inline void thetas(const Matrix_t &new_values) {
            if (new_values.rows() != N_) {
                throw std::domain_error("ERROR: thetas size should be equal to N (prediction horizon)");
            }   
            thetas_ = new_values;
        }
    protected:
        Matrix_t thetas_;
        Matrix_t Hp_;
};

class MPCLinearHomography : public MPCLinearVisualConstraint {
    public:
        MPCLinearHomography(std::string type, int N);
        inline Matrix_t A() const override {
            VisualConstraintEntryInterface *linear_entry_hxx_ = dynamic_cast<VisualConstraintEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!linear_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not VisualConstraintEntryInterface*");
            return linear_entry_hxx_->a() * Matrix_t::Identity(N_, N_);
        }
        inline Matrix_t B() const override {
            VisualConstraintEntryInterface *linear_entry_hxx_ = dynamic_cast<VisualConstraintEntryInterface *>(vconstraint_entry_hxx_.get());
             if (!linear_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not VisualConstraintEntryInterface*");
            return linear_entry_hxx_->b() * Matrix_t::Identity(N_, N_);
        }
        inline Matrix_t C() const override {
            Matrix_t C(N_, 1);
            C.setZero();
            VisualConstraintEntryInterface *linear_entry_hxx_ = dynamic_cast<VisualConstraintEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!linear_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not VisualConstraintEntryInterface*");
            for (int i = 0; i < N_; i++) {
                linear_entry_hxx_->theta(thetas_(i,0)); 
                C(i, 0) = linear_entry_hxx_->c();
            }
            return C;
        }
        inline void d(FPTYPE new_value) {
            HomographyEntryInterface *homography_entry_hxx_ = dynamic_cast<HomographyEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!homography_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not HomographyEntryInterface*");
            homography_entry_hxx_->d(new_value);
        }       
        inline void nz(FPTYPE new_value) {
            HomographyEntryInterface *homography_entry_hxx_ = dynamic_cast<HomographyEntryInterface *>(vconstraint_entry_hxx_.get());            
            if (!homography_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not HomographyEntryInterface*");
            homography_entry_hxx_->nz(new_value);
        }
        inline void nx(FPTYPE new_value) {
            HomographyEntryInterface *homography_entry_hxx_ = dynamic_cast<HomographyEntryInterface *>(vconstraint_entry_hxx_.get());            
            if (!homography_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not HomographyEntryInterface*");
            homography_entry_hxx_->nx(new_value);
        }
        inline void ny(FPTYPE new_value) {
            HomographyEntryInterface *homography_entry_hxx_ = dynamic_cast<HomographyEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!homography_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not HomographyEntryInterface*");
            homography_entry_hxx_->ny(new_value);
        }
};


class MPCLinearEssential : public MPCLinearVisualConstraint {
    public:
        MPCLinearEssential(std::string type, int N);
        inline Matrix_t A() const override {
            VisualConstraintEntryInterface *linear_entry_hxx_ = dynamic_cast<VisualConstraintEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!linear_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not VisualConstraintEntryInterface*");
            Matrix_t A(N_, N_);
            A.setZero();
            for (int i = 0; i < N_; i++) {
                linear_entry_hxx_->theta(thetas_(i,0)); 
                A(i, i) = linear_entry_hxx_->a();
            }
            return A;
        }
        inline Matrix_t B() const override {
            VisualConstraintEntryInterface *linear_entry_hxx_ = dynamic_cast<VisualConstraintEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!linear_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not VisualConstraintEntryInterface*");
            Matrix_t B(N_, N_);
            B.setZero();
            for (int i = 0; i < N_; i++) {
                linear_entry_hxx_->theta(thetas_(i,0)); 
                B(i, i) = linear_entry_hxx_->b();
            }
            return B;
        }
        inline Matrix_t C() const override {
            VisualConstraintEntryInterface *linear_entry_hxx_ = dynamic_cast<VisualConstraintEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!linear_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not VisualConstraintEntryInterface*");
            Matrix_t C(N_, 1);
            C.setZero();
            for (int i = 0; i < N_; i++) {
                linear_entry_hxx_->theta(thetas_(i,0)); 
                C(i, 0) = linear_entry_hxx_->c();
            }
            return C;
        }
        inline void ck_y_t(FPTYPE new_value) {
            EssentialEntryInterface *ess_entry_hxx_ = dynamic_cast<EssentialEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!ess_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not EssentialEntryInterface*");
            ess_entry_hxx_->ck_y_t(new_value);
        }
        inline void y_real(FPTYPE new_value) {
            EssentialEntryInterface *ess_entry_hxx_ = dynamic_cast<EssentialEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!ess_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not EssentialEntryInterface*");
            ess_entry_hxx_->y_real(new_value);
        }
};

class MPCNonLinearEssential : public MPCVisualConstraint {
    public:
        MPCNonLinearEssential(std::string type, int N);
        inline Matrix_t HX(const Matrix_t &phis) const {
            EssentialEntryInterface *ess_entry_hxx_ = dynamic_cast<EssentialEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!ess_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not EssentialEntryInterface*");
            return ess_entry_hxx_->GetHX(phis);
        }
        inline Matrix_t HY(const Matrix_t &phis) const {
            EssentialEntryInterface *ess_entry_hxx_ = dynamic_cast<EssentialEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!ess_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not EssentialEntryInterface*");
            return ess_entry_hxx_->GetHY(phis);
        }
        inline Matrix_t HT(const Matrix_t &Xk1,
                           const Matrix_t &Yk1,
                           const Matrix_t &phis) const {
            EssentialEntryInterface *ess_entry_hxx_ = dynamic_cast<EssentialEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!ess_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not EssentialEntryInterface*");
            return ess_entry_hxx_->GetHT(Xk1,Yk1,phis);
        }
        inline Matrix_t RE(const Matrix_t &Xk1,
                           const Matrix_t &Yk1,
                           const Matrix_t &phis) const {
            EssentialEntryInterface *ess_entry_hxx_ = dynamic_cast<EssentialEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!ess_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not EssentialEntryInterface*");
            return ess_entry_hxx_->GetRE(Xk1,Yk1,phis);
        }
        inline void ck_y_t(FPTYPE new_value) {
            EssentialEntryInterface *ess_entry_hxx_ = dynamic_cast<EssentialEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!ess_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not EssentialEntryInterface*");
            ess_entry_hxx_->ck_y_t(new_value);
        }
        inline void y_real(FPTYPE new_value) {
            EssentialEntryInterface *ess_entry_hxx_ = dynamic_cast<EssentialEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!ess_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not EssentialEntryInterface*");
            ess_entry_hxx_->y_real(new_value);
        }
};

class MPCNonLinearHomography : public MPCVisualConstraint {
    public:
        MPCNonLinearHomography(std::string type, int N);
        inline Matrix_t HX(const Matrix_t &thetas) const {
            HomographyEntryInterface *hom_entry_hxx_ = dynamic_cast<HomographyEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!hom_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not HomographyEntryInterface*");
            return hom_entry_hxx_->GetHX(thetas);
        }
        inline Matrix_t HY(const Matrix_t &thetas) const {
            HomographyEntryInterface *hom_entry_hxx_ = dynamic_cast<HomographyEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!hom_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not HomographyEntryInterface*");
            return hom_entry_hxx_->GetHY(thetas);
        }
        inline Matrix_t HT(const Matrix_t &Xk1,
                           const Matrix_t &Yk1,
                           const Matrix_t &thetas) const {
            HomographyEntryInterface *hom_entry_hxx_ = dynamic_cast<HomographyEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!hom_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not HomographyEntryInterface*");
            return hom_entry_hxx_->GetHT(Xk1,Yk1,thetas);
        }
        inline Matrix_t RE(const Matrix_t &Xk1,
                           const Matrix_t &Yk1,
                           const Matrix_t &phis) const {
            HomographyEntryInterface *hom_entry_hxx_ = dynamic_cast<HomographyEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!hom_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not HomographyEntryInterface*");
            return hom_entry_hxx_->GetRE(Xk1,Yk1,phis);
        }
        inline void d(FPTYPE new_value) {
            HomographyEntryInterface *homography_entry_hxx_ = dynamic_cast<HomographyEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!homography_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not HomographyEntryInterface*");
            homography_entry_hxx_->d(new_value);
        }
        inline void nz(FPTYPE new_value) {
            HomographyEntryInterface *homography_entry_hxx_ = dynamic_cast<HomographyEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!homography_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not HomographyEntryInterface*");
            homography_entry_hxx_->nz(new_value);
        }
        inline void nx(FPTYPE new_value) {
            HomographyEntryInterface *homography_entry_hxx_ = dynamic_cast<HomographyEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!homography_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not HomographyEntryInterface*");
            homography_entry_hxx_->nx(new_value);
        }

        inline void ny(FPTYPE new_value) {
            HomographyEntryInterface *homography_entry_hxx_ = dynamic_cast<HomographyEntryInterface *>(vconstraint_entry_hxx_.get());
            if (!homography_entry_hxx_)
                throw std::runtime_error("ERROR: the vconstraint_entry_hxx_ is not HomographyEntryInterface*");
            homography_entry_hxx_->ny(new_value);
        }

};
/******************************************************************************
 * Parameters:
 *   o max_step_len         Maximal distance between centers of the feet along the sagittal plane (meter)
 *   o min_feet_dist        Maximal distance between centers of the feet along the coronal plane (meter)
 *   o max_feet_dist        Maximal distance between centers of the feet along the coronal plane (meter)
 *   o feet_dist_default    The default distance between the feet in initial / final double supports (meter)
 *   o foot_len             Length of a foot (meter)
 *   o foot_width           Width of a foot (meter)
 *   o com_height           center of mass height
 *****************************************************************************/

class RobotModelInterface {
    public:
        virtual FPTYPE max_step_len() = 0;
        virtual FPTYPE min_feet_dist() = 0;
        virtual FPTYPE max_feet_dist() = 0;
        virtual FPTYPE feet_dist_default() = 0;
        virtual FPTYPE foot_len() = 0;
        virtual FPTYPE foot_width() = 0;
        virtual FPTYPE com_height() = 0;
        virtual FPTYPE camera_position() = 0;
        virtual ~RobotModelInterface() { }
};


RobotModelInterface* InstantiateRobotModel(std::string name);

class NaoModel : public RobotModelInterface {
    public:
        FPTYPE max_step_len() { return 0.10; }
        FPTYPE min_feet_dist() { return 0.10; }
        FPTYPE max_feet_dist()  {return 0.15; }
        FPTYPE feet_dist_default() {return 0.10; }
        FPTYPE foot_len() { return 0.1372; }
        FPTYPE foot_width() { return 0.058; }
        FPTYPE camera_position() { return 0.452; };
        FPTYPE com_height() { return 0.262; }
};


class HRPModel : public RobotModelInterface {
    public:
        FPTYPE max_step_len() { return 0.25; }
        FPTYPE min_feet_dist() { return 0.19; }
        FPTYPE max_feet_dist()  {return 0.29; }
        FPTYPE feet_dist_default() {return 0.19; }
        FPTYPE foot_len() { return 0.25; }
        FPTYPE foot_width() { return 0.14; }
        FPTYPE camera_position() { return 1.061; };
        FPTYPE com_height() { return 0.711; }
};

class JVRCModel : public RobotModelInterface {
    public:
        FPTYPE max_step_len() { return 0.2; }
        FPTYPE min_feet_dist() { return 0.15; }
        FPTYPE max_feet_dist()  {return 0.20; }
        FPTYPE feet_dist_default() {return 0.19; }
        FPTYPE foot_len() { return 0.22; }
        FPTYPE foot_width() { return 0.10; }
        FPTYPE camera_position() { return 1.585; };
        FPTYPE com_height() { return 0.85; }
};

#endif
