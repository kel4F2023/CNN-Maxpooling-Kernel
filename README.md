# Max Pooling Kernel

### Compilaiton

To make the executable, do `make compile`

the exeutables are

- performance.x for running perf statistics and mid-performance.x for runing perf stats for old implementation.

- driver.x for test runs

- experiment.x for running from varying k values and mid-experiment.x for old implementation.


### Test correctness
The driver.c file contains a test run where you can modify the macro VERBOSE 0 and CORRECTNESS_CHECK to print more information

To run a test, do `make test`, and you can adjust what to print out by modifying the macros.

### Run experiment
The experiment.c file contains the code to run from varying k values. 

To run a experiment, do `make run`

### Performance results 
The experiment.c file contains the code to run the kernel only, no packing and correctness

To run a experiment, do `make perf`


### Reproduce results
To reproduce the data,

run `make init-data` to start over

run `make data` to build the dataset with more data

### Visualization

The visualizaitons are in the ipython notebook named `visual.ipynb`
