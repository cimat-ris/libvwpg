#!/usr/bin/env python

## Licensed to the Apache Software Foundation (ASF) under one
## or more contributor license agreements.  See the NOTICE file
## distributed with this work for additional information
## regarding copyright ownership.  The ASF licenses this file
## to you under the Apache License, Version 2.0 (the
## "License"); you may not use this file except in compliance
## with the License.  You may obtain a copy of the License at
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing,
## software distributed under the License is distributed on an
## "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
## KIND, either express or implied.  See the License for the
## specific language governing permissions and limitations
## under the License.

import sys
import math
import os

sys.path.append("..") #con esto me salgo de la carpeta python_bindings
from build.QPHomograpy_py import QPHomographySimLinear 

def main():
    """ main function """
    if len(sys.argv) < 2:
        print ("Usage:")
        print ("    python {0} <options> <ini>\n".format(sys.argv[0]))
        sys.exit(-1)

    for ini_file in sys.argv[1:]:
        print ("Analyzing {0}...".format(ini_file))
        try:
            print(ini_file)
            sim = QPHomographySimLinear(ini_file)

            while sim.Continue():
                sim.LogCurrentResults()
                sim.Update()
                sim.SolveProblem()
        except IOError:
            print ("ERROR: Invalid ini file '{0}'".format(ini_file))
            sys.exit(-1)


if __name__ == '__main__':
    main()
