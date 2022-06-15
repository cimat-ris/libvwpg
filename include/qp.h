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

#ifndef QP_H
#define QP_H

#include <string>

#include <Eigen/Dense>

#include "qpOASES.hpp"

#include "feet.h"


/*
 * Interface all formulations should implement to be able to use the QPSolver.
 * The problem should be expressed in the following canonical form:
 *
 *    min w.r.t. x  (1/2)(x^T)Qx + (x^T)g
 *
 *    s.t.
 *      lbA <= Ax <= ubA
 *      lb <= x <= ub
 *
 */

class QProblemInterface {

    public:
        virtual ~QProblemInterface(){};
        virtual Matrix_t GetH() = 0;
        virtual Matrix_t Getg() = 0;
        virtual Matrix_t GetA() = 0;
        virtual Matrix_t GetlbA() = 0;
        virtual Matrix_t GetubA() = 0;
        virtual Matrix_t Getlb() = 0;
        virtual Matrix_t Getub() = 0;
        virtual int GetNumberOfVariables() = 0;
        virtual int GetNumberOfConstraints() = 0;
};



class QPSolver {

    public:
        QPSolver(QProblemInterface *qp_problem, int mWSR=100000);
        void SolveProblem();
        Matrix_t GetSolution();
        FPTYPE GetObjectiveFunctionValue();

    private:
        QProblemInterface* problem_;
        bool is_first_solution_;
        int nWSR_;
        qpOASES::SQProblem qpoases_solver_;
        qpOASES::returnValue return_value_;

    private:
        void LogError(const std::string &message, qpOASES::returnValue error);
        std::string ErrorToString(int error);
};


#endif
