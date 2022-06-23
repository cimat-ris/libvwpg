# libvwpg

**Visual Walking Pattern Generation** library, that uses homography based control for the walking pattern generation of a humanoid robot.

In order to compile this code two external dependencies should be installed: eigen and 

First, compile qpOases with CMake:

```bash
> cd qpOASES/
> mkdir bin
> make
```

Then compile the libvwpg library. From the libvwpg directory:

```bash
> mkdir build
> cd build
> cmake ..
> make
```

To run the test file:

```bash
> ./bin/test1 <file.ini>
```

INI files are used to define the parameters for the simulation. There are multiple
files included in this package in the working_configs directory.

Once the execution runs, you should
redirect the standard output to a .txt file.

```bash
> ./bin/test1 <file.ini>    > <log_file.txt>
```

Then, to plot the results use the plot_results.py script as follows:

```bash
> python ../tools/plot_results.py <log_file.txt>
```


To run the  example  QPHomoTest.py of python:

```bash
>python QPHomoTest.py <file.ini>
```

