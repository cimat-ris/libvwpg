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

#include "QPSimulators.h"


//******************************************************************************
//
//                        QPHomographySimLinear
//
//******************************************************************************

QPHomographySimLinear::QPHomographySimLinear(const std::string nameFileParameters):
    simulator_(nameFileParameters),
    qp_solver_(dynamic_cast<QProblemInterface*>(&simulator_))
{
}

FPTYPE QPHomographySimLinear::GetCoMAbsoluteX() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return X(position);
}

FPTYPE QPHomographySimLinear::GetCoMAbsoluteY() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return Y(position);
}

FPTYPE QPHomographySimLinear::GetCoMAbsoluteZ() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return Z(position);
}

FPTYPE QPHomographySimLinear::GetCoMAbsoluteTheta(){
    return simulator_.GetCoMAbsoluteTheta();
}

FPTYPE QPHomographySimLinear::GetSupportFootAbsoluteX(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return X(position);
}

FPTYPE QPHomographySimLinear::GetSupportFootAbsoluteY(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return Y(position);
}

FPTYPE QPHomographySimLinear::GetSupportFootAbsoluteZ(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return Z(position);
}

FPTYPE QPHomographySimLinear::GetSupportFootAbsoluteTheta(){
    return simulator_.GetSupportFootAbsoluteTheta();
}

FPTYPE QPHomographySimLinear::GetNextSupportFootAbsoluteX() {
    auto position = simulator_.GetNextSupportFootAbsolutePosition(qp_solver_.GetSolution());
    return X(position);
}

FPTYPE QPHomographySimLinear::GetNextSupportFootAbsoluteY() {
    auto position = simulator_.GetNextSupportFootAbsolutePosition(qp_solver_.GetSolution());
    return Y(position);
}

void QPHomographySimLinear::SolveProblem(){
    qp_solver_.SolveProblem();
}

bool QPHomographySimLinear::Continue(){
    return simulator_.Continue();
}

void QPHomographySimLinear::Update(){
    simulator_.Update(qp_solver_.GetSolution());
}

void QPHomographySimLinear::LogCurrentResults(){
    simulator_.LogCurrentResults(qp_solver_.GetSolution());
}

FPTYPE QPHomographySimLinear::GetNextSupportFootAbsoluteTheta(){
    return simulator_.GetNextSupportFootAbsoluteTheta();
}

void QPHomographySimLinear::SetCoMValues(const FPTYPE x_position, const FPTYPE y_position, const FPTYPE z_position, const FPTYPE x_speed, const FPTYPE y_speed ){
    simulator_.SetCoMValues(x_position,y_position,z_position,x_speed, y_speed );
}

void  QPHomographySimLinear::GetObjectiveFunctionValue(){
    common::logString("[objective_function]: " + boost::lexical_cast<std::string>(qp_solver_.GetObjectiveFunctionValue()));
}


//******************************************************************************
//
//                        QPHomographyRealLinear
//
//******************************************************************************

QPHomographyRealLinear::QPHomographyRealLinear(const std::string nameFileParameters, const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                                               const FPTYPE h21, const FPTYPE h22, const FPTYPE h23,
                                               const FPTYPE h31, const FPTYPE h32, const FPTYPE h33):
        simulator_(nameFileParameters,h11,h12,h13,h21,h22,h23,h31,h32,h33),
        qp_solver_(dynamic_cast<QProblemInterface*>(&simulator_))
{
}

FPTYPE QPHomographyRealLinear::GetCoMAbsoluteX() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return X(position);
}

FPTYPE QPHomographyRealLinear::GetCoMAbsoluteY() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return Y(position);
}

FPTYPE QPHomographyRealLinear::GetCoMAbsoluteZ() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return Z(position);
}

FPTYPE QPHomographyRealLinear::GetCoMAbsoluteTheta(){
    return simulator_.GetCoMAbsoluteTheta();
}

FPTYPE QPHomographyRealLinear::GetSupportFootAbsoluteX(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return X(position);
}

FPTYPE QPHomographyRealLinear::GetSupportFootAbsoluteY(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return Y(position);
}

FPTYPE QPHomographyRealLinear::GetSupportFootAbsoluteZ(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return Z(position);
}

FPTYPE QPHomographyRealLinear::GetSupportFootAbsoluteTheta(){
    return simulator_.GetSupportFootAbsoluteTheta();
}

FPTYPE QPHomographyRealLinear::GetNextSupportFootAbsoluteX() {
    auto position = simulator_.GetNextSupportFootAbsolutePosition(qp_solver_.GetSolution());
    return X(position);
}

FPTYPE QPHomographyRealLinear::GetNextSupportFootAbsoluteY() {
    auto position = simulator_.GetNextSupportFootAbsolutePosition(qp_solver_.GetSolution());
    return Y(position);
}

FPTYPE QPHomographyRealLinear::GetNextSupportFootAbsoluteTheta(){
    return simulator_.GetNextSupportFootAbsoluteTheta();
}

void QPHomographyRealLinear::SolveProblem(){
    qp_solver_.SolveProblem();
}

bool QPHomographyRealLinear::Continue(){
    return simulator_.Continue();
}

void QPHomographyRealLinear::Update(){
    simulator_.Update(qp_solver_.GetSolution());
}

void QPHomographyRealLinear::LogCurrentResults(){
    simulator_.LogCurrentResults(qp_solver_.GetSolution());
}

void QPHomographyRealLinear::SetCurrentHomography(const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                                                  const FPTYPE h21, const FPTYPE h22, const FPTYPE h23,
                                                  const FPTYPE h31, const FPTYPE h32, const FPTYPE h33,
                                                  bool isComputeHomography) {

    simulator_.SetCurrentHomography(h11,h12,h13,h21,h22,h23,h31,h32,h33,isComputeHomography);
}

void QPHomographyRealLinear::SetCoMValues(const FPTYPE x_position, const FPTYPE y_position, const FPTYPE z_position, const FPTYPE x_speed, const FPTYPE y_speed ){
    simulator_.SetCoMValues(x_position,y_position,z_position,x_speed,y_speed );
}

void  QPHomographyRealLinear::GetObjectiveFunctionValue(){
    common::logString("[objective_function]: " + boost::lexical_cast<std::string>(qp_solver_.GetObjectiveFunctionValue()));
}


//******************************************************************************
//
//                        QPHomographyObstaclesLinear
//
//******************************************************************************

QPHomographyObstaclesLinear::QPHomographyObstaclesLinear(const std::string nameFileParameters, const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                                                        const FPTYPE h21, const FPTYPE h23, const FPTYPE h31, const FPTYPE h32, const FPTYPE h33, 
                                                        const FPTYPE c1,  const FPTYPE c2,  const FPTYPE c3, const bool isSameSide):
    simulator_(nameFileParameters,h11,h12,h13,h21,h23,h31,h32,h33,c1,c2,c3,isSameSide),
    qp_solver_(dynamic_cast<QProblemInterface*>(&simulator_))
{
}

FPTYPE QPHomographyObstaclesLinear::GetCoMAbsoluteX() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return X(position);
}

FPTYPE QPHomographyObstaclesLinear::GetCoMAbsoluteY() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return Y(position);
}

FPTYPE QPHomographyObstaclesLinear::GetCoMAbsoluteZ() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return Z(position);
}

FPTYPE QPHomographyObstaclesLinear::GetCoMAbsoluteTheta(){
    return simulator_.GetCoMAbsoluteTheta();
}

FPTYPE QPHomographyObstaclesLinear::GetSupportFootAbsoluteX(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return X(position);
}

FPTYPE QPHomographyObstaclesLinear::GetSupportFootAbsoluteY(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return Y(position);
}

FPTYPE QPHomographyObstaclesLinear::GetSupportFootAbsoluteZ(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return Z(position);
}

FPTYPE QPHomographyObstaclesLinear::GetSupportFootAbsoluteTheta(){
    return simulator_.GetSupportFootAbsoluteTheta();
}

FPTYPE QPHomographyObstaclesLinear::GetNextSupportFootAbsoluteX() {
    auto position = simulator_.GetNextSupportFootAbsolutePosition(qp_solver_.GetSolution());
    return X(position);
}

FPTYPE QPHomographyObstaclesLinear::GetNextSupportFootAbsoluteY() {
    auto position = simulator_.GetNextSupportFootAbsolutePosition(qp_solver_.GetSolution());
    return Y(position);
}

FPTYPE QPHomographyObstaclesLinear::GetNextSupportFootAbsoluteTheta(){
    return simulator_.GetNextSupportFootAbsoluteTheta();
}

void QPHomographyObstaclesLinear::SolveProblem(){
    qp_solver_.SolveProblem();
}

bool QPHomographyObstaclesLinear::Continue(){
    return simulator_.Continue();
}

void QPHomographyObstaclesLinear::Update(){
    simulator_.Update(qp_solver_.GetSolution());
}

void QPHomographyObstaclesLinear::LogCurrentResults(){
    simulator_.LogCurrentResults(qp_solver_.GetSolution());
}

void QPHomographyObstaclesLinear::SetCurrentHomography( const FPTYPE h11, const FPTYPE h12, const FPTYPE h13,
                                   const FPTYPE h21, const FPTYPE h23,
                                   const FPTYPE h31, const FPTYPE h32, const FPTYPE h33, const bool isObstacle, const bool isComputeHomography,
                                   const FPTYPE c1,  const FPTYPE c2,  const FPTYPE c3, const int iteration){

    simulator_.SetCurrentHomography(h11,h12,h13,h21,h23,h31,h32,h33,isObstacle,isComputeHomography,c1,c2,c3,iteration);
}

void QPHomographyObstaclesLinear::SetCoMValues(const FPTYPE x_position, const FPTYPE y_position, const FPTYPE z_position, const FPTYPE x_speed, const FPTYPE y_speed ){
    simulator_.SetCoMValues(x_position,y_position,z_position,x_speed,y_speed );
}

void  QPHomographyObstaclesLinear::GetObjectiveFunctionValue(){
    common::logString("[objective_function]: " + boost::lexical_cast<std::string>(qp_solver_.GetObjectiveFunctionValue()));
}

bool QPHomographyObstaclesLinear::IsSupportFootChange(){
    return simulator_.IsSupportFootChange();
}

//******************************************************************************
//
//                        QPHerdtSim
//
//******************************************************************************

QPHerdtSim::QPHerdtSim(const std::string nameFileParameters):
    simulator_(nameFileParameters),
    qp_solver_(dynamic_cast<QProblemInterface*>(&simulator_))
{
}


FPTYPE QPHerdtSim::GetCoMAbsoluteX() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return X(position);
}

FPTYPE QPHerdtSim::GetCoMAbsoluteY() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return Y(position);
}

FPTYPE QPHerdtSim::GetCoMAbsoluteZ() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return Z(position);
}

FPTYPE QPHerdtSim::GetCoMAbsoluteTheta(){
    return simulator_.GetCoMAbsoluteTheta();
}

FPTYPE QPHerdtSim::GetSupportFootAbsoluteX(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return X(position);
}

FPTYPE QPHerdtSim::GetSupportFootAbsoluteY(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return Y(position);
}

FPTYPE QPHerdtSim::GetSupportFootAbsoluteZ(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return Z(position);
}

FPTYPE QPHerdtSim::GetSupportFootAbsoluteTheta(){
    return simulator_.GetSupportFootAbsoluteTheta();
}

FPTYPE QPHerdtSim::GetNextSupportFootAbsoluteX() {
    auto position = simulator_.GetNextSupportFootAbsolutePosition(qp_solver_.GetSolution());
    return X(position);
}

FPTYPE QPHerdtSim::GetNextSupportFootAbsoluteY() {
    auto position = simulator_.GetNextSupportFootAbsolutePosition(qp_solver_.GetSolution());
    return Y(position);
}

void QPHerdtSim::SolveProblem(){
    qp_solver_.SolveProblem();
}

bool QPHerdtSim::Continue(){
    return simulator_.Continue();
}

void QPHerdtSim::Update(){
    simulator_.Update(qp_solver_.GetSolution());
}

void QPHerdtSim::LogCurrentResults(){
    simulator_.LogCurrentResults(qp_solver_.GetSolution());
}

FPTYPE QPHerdtSim::GetNextSupportFootAbsoluteTheta(){
    return simulator_.GetNextSupportFootAbsoluteTheta();
}

void QPHerdtSim::SetCoMValues(const FPTYPE x_position, const FPTYPE y_position, const FPTYPE z_position, const FPTYPE x_speed, const FPTYPE y_speed ){
    simulator_.SetCoMValues(x_position,y_position,z_position,x_speed, y_speed );
}

void  QPHerdtSim::GetObjectiveFunctionValue(){
    common::logString("[objective_function]: " + boost::lexical_cast<std::string>(qp_solver_.GetObjectiveFunctionValue()));
}

//******************************************************************************
//
//                        QPHerdtReal
//
//******************************************************************************

QPHerdtReal::QPHerdtReal(const std::string nameFileParameters, const FPTYPE x_speed_ref, const FPTYPE y_speed_ref, const FPTYPE theta_ref):
    simulator_(nameFileParameters,x_speed_ref,y_speed_ref,theta_ref),
    qp_solver_(dynamic_cast<QProblemInterface*>(&simulator_))
{
}

FPTYPE QPHerdtReal::GetCoMAbsoluteX() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return X(position);
}

FPTYPE QPHerdtReal::GetCoMAbsoluteY() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return Y(position);
}

FPTYPE QPHerdtReal::GetCoMAbsoluteZ() {
    auto position = simulator_.GetCoMAbsolutePosition();
    return Z(position);
}

FPTYPE QPHerdtReal::GetCoMAbsoluteTheta(){
    return simulator_.GetCoMAbsoluteTheta();
}

FPTYPE QPHerdtReal::GetSupportFootAbsoluteX(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return X(position);
}

FPTYPE QPHerdtReal::GetSupportFootAbsoluteY(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return Y(position);
}

FPTYPE QPHerdtReal::GetSupportFootAbsoluteZ(){
    auto position = simulator_.GetSupportFootAbsolutePosition();
    return Z(position);
}

FPTYPE QPHerdtReal::GetSupportFootAbsoluteTheta(){
    return simulator_.GetSupportFootAbsoluteTheta();
}

FPTYPE QPHerdtReal::GetNextSupportFootAbsoluteX() {
    auto position = simulator_.GetNextSupportFootAbsolutePosition(qp_solver_.GetSolution());
    return X(position);
}

FPTYPE QPHerdtReal::GetNextSupportFootAbsoluteY() {
    auto position = simulator_.GetNextSupportFootAbsolutePosition(qp_solver_.GetSolution());
    return Y(position);
}

void QPHerdtReal::SolveProblem(){
    qp_solver_.SolveProblem();
}

bool QPHerdtReal::Continue(){
    return simulator_.Continue();
}

void QPHerdtReal::Update(){
    simulator_.Update(qp_solver_.GetSolution());
}

void QPHerdtReal::LogCurrentResults(){
    simulator_.LogCurrentResults(qp_solver_.GetSolution());
}

FPTYPE QPHerdtReal::GetNextSupportFootAbsoluteTheta(){
    return simulator_.GetNextSupportFootAbsoluteTheta();
}

void QPHerdtReal::SetCoMValues(const FPTYPE x_position, const FPTYPE y_position, const FPTYPE z_position, const FPTYPE x_speed, const FPTYPE y_speed ){
    simulator_.SetCoMValues(x_position,y_position,z_position,x_speed, y_speed );
}

void  QPHerdtReal::GetObjectiveFunctionValue(){
    common::logString("[objective_function]: " + boost::lexical_cast<std::string>(qp_solver_.GetObjectiveFunctionValue()));
}

void QPHerdtReal::SetCurrentReferenceSpeed(const FPTYPE x_speed_ref, const FPTYPE y_speed_ref, const FPTYPE theta_ref){
    simulator_.SetCurrentReferenceSpeed( x_speed_ref, y_speed_ref, theta_ref);
}