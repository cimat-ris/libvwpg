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

#include <iostream>
#include <exception>
#include <string>
#include <map>

#include <Eigen/Dense>

#include "qpOASES.hpp"

#include "qp.h"
#include "common.h"


QPSolver::QPSolver(QProblemInterface *qp_problem, long nWSR) :
    problem_(qp_problem),
    nWSR_(nWSR),
    is_first_solution_(true),
    qpoases_solver_(problem_->GetNumberOfVariables(),problem_->GetNumberOfConstraints())
{
    long tmp_nWSR = nWSR_;   // qpOASES overrides the value of nWSR, use a copy
    qpOASES::returnValue return_value;
    qpoases_solver_.setPrintLevel(qpOASES::PL_NONE);
    return_value = qpoases_solver_.init(problem_->GetH().data(),
                                        problem_->Getg().data(),
                                        problem_->GetA().data(),
                                        problem_->Getlb().data(),
                                        problem_->Getub().data(),
                                        problem_->GetlbA().data(),
                                        problem_->GetubA().data(),
                                        tmp_nWSR);
    if (return_value != qpOASES::SUCCESSFUL_RETURN) {
        LogError("ERROR: qpOASES unable to initialize QP problem: ", return_value);
        throw std::exception();
    }

}


void QPSolver::SolveProblem()
{

    if (!is_first_solution_) {
        long tmp_nWSR = nWSR_;   // qpOASES overrides the value of nWSR, use a copy
        qpOASES::returnValue return_value;

        return_value = qpoases_solver_.hotstart(problem_->GetH().data(),
                                                problem_->Getg().data(),
                                                problem_->GetA().data(),
                                                problem_->Getlb().data(),
                                                problem_->Getub().data(),
                                                problem_->GetlbA().data(),
                                                problem_->GetubA().data(),
                                                tmp_nWSR);

        if (return_value != qpOASES::SUCCESSFUL_RETURN) {
            LogError("ERROR: qpOASES unable to solve QP problem: ", return_value);
            throw std::exception();
        }

    }
}


Matrix_t QPSolver::GetSolution()
{
    if (is_first_solution_)
        is_first_solution_ = false;
    Matrix_t solution(problem_->GetNumberOfVariables(), 1);
    qpoases_solver_.getPrimalSolution(solution.data());
    return solution;
}


FPTYPE QPSolver::GetObjectiveFunctionValue()
{
    return qpoases_solver_.getObjVal();
}

// private:


std::string QPSolver::ErrorToString(int error)
{
  switch (error) {
  case -3:
    return "QP is unbounded and thus could not be solved";
  case -2:
    return "QP is infeasible and thus could not be solved";
  case -1:
    return "QP could not be solved due to an internal error";
  case 1:
    return "QP could not be solved within the given number of iterations";
  default:
    return "Unknown QP error";
  }
}


inline void QPSolver::LogError(const std::string &message, qpOASES::returnValue error)
{
    std::cerr <<  message << ErrorToString(qpOASES::getSimpleStatus(error)) << std::endl;
}
