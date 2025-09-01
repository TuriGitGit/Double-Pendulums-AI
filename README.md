# Double-Pendulums

Generate a dataset of hundreds of millions of chaotic double pendulum simulations and train a Neural Network to predict where the pendulum will be


# How-to

## Run Simulations

### Linux
Open Double-Pendulums in Terminal
- 1 `export OMP_NUM_THREADS={threads}`
- 2 `gcc -O3 -march=native -ffast-math -fno-unsafe-math-optimizations -funroll-loops -fno-trapping-math -fno-signaling-nans -fopenmp Simulations.c -lm -o simulations`
- 3 `./simulations`

### Windows
*note: I have not officialy tested this so make an issue if it does not work. I will remove this once i have tested it*

Open Double-Pendulums in Powershell / Windows Terminal **DO NOT USE COMMAND PROMPT**
- 1 install gcc if it is not already installed https://www.google.com/search?q=how+to+install+gcc+on+windows you can check if it is installed with `gcc --version`
- 2 compile with `gcc -O3 -march=native -ffast-math -fno-unsafe-math-optimizations -funroll-loops -fno-trapping-math -fno-signaling-nans -fopenmp Simulations.c -lm -o simulations`
- 3 set threads to use with `$env:OMP_NUM_THREADS={threads}` replace `{threads}` with your desired thread count
- 4 run with `./simulations.exe`


### MacOS support will come later bc apple sucks >:[
*note: I do not own any apple devices nor plan to, so mac support may take a while*
