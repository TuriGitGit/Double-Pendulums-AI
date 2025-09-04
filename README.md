# Double-Pendulums-AI
Generate a dataset of hundreds of millions of chaotic double pendulum simulations at various angles, lengths, and simulation times.
Then use Deep Learning with pytorch train a Neural Network to predict where the pendulums will be.


# How-to

## 1. Run Simulations
![WARNING](https://img.shields.io/badge/WARNING:-red) I highly suggest reading through [Simulations.c](Simulations.c) to make sure variables like `STEPS`, `SIMS`, and `buffer_size` are tuned to your computer's hardware capabilities

  ### Linux
  Open Double-Pendulums-AI in a single Terminal
  - 1 `export OMP_NUM_THREADS={threads}`
  - 2 `gcc -O3 -march=native -ffast-math -fno-unsafe-math-optimizations -funroll-loops -fno-trapping-math -fno-signaling-nans -fopenmp Simulations.c -lm -o simulations`
  - 3 `./simulations`
  
  ### Windows
  ![NOTE](https://img.shields.io/badge/NOTE:-orange) *I have not officially tested this so make an issue if it does not work. I will remove this once i have tested it*
  
  Open Double-Pendulums-AI in a single Powershell / Windows Terminal **DO NOT USE COMMAND PROMPT**
  - install gcc if it is not already installed: https://www.google.com/search?q=how+to+install+gcc+on+windows you can check if it is installed with `gcc --version`
  - 1 compile with `gcc -O3 -march=native -ffast-math -fno-unsafe-math-optimizations -funroll-loops -fno-trapping-math -fno-signaling-nans -fopenmp Simulations.c -lm -o simulations`
  - 2 set threads to use with `$env:OMP_NUM_THREADS=<threads>` replace `<threads>` with your desired thread count.
  - 3 run with `./simulations.exe`
  
  
  ### MacOS support will come later bc apple sucks >:[
  ![NOTE](https://img.shields.io/badge/NOTE:-orange) *I do not own any apple devices nor plan to, so mac support may take a while*

## 2. Run Training

  ### Linux
  Open a single terminal
  If you already have a pip venv active you can skip to step 3.
  - 1 `python3 -m venv ~/p1p`
  - 2 `source ~/p1p/bin/activate`
  - 3 `pip install --upgrade pip`
  - 4 `pip install numpy wandb torch`
  - 5(only for IDE users) in your IDE of choice set the python env variable to `~/p1p/bin/python` or your own pip venv if you already have one.
  - 6 `python3 CSVtoBinary.py` or run with the IDE of your choice.
  - 7 `python3 Train.py` or run with the IDE of your choice.
  
  ### Windows
  ![NOTE](https://img.shields.io/badge/NOTE:-orange) *I have not officially tested this so make an issue if it does not work. I will remove this once i have tested it*
  
  Open a single powershell terminal
  If you already have a pip venv active you can skip to step 3
  - install python if you do not already have it installed: https://www.python.org/downloads/windows/
  - 1 `python -m venv C:\Users\<YourUsername>\p1p` replace `<YourUsername>` with your windows username.
  - 2 `C:\Users\<YourUsername>\p1p\Scripts\Activate.ps1` replace `<YourUsername>` with your windows username.
  - 3 `python -m pip install --upgrade pip`
  - 4 `python -m pip install numpy wandb torch`
  - 5(only for IDE users) in your IDE of choice set the python env variable to `C:\Users\<YourUsername>\p1p\Scripts\python.exe` replace `<YourUsername>` with your windows username.
  - 6 `python CSVtoBinary.py` or run with the IDE of your choice.
  - 7 `python Train.py` or run with the IDE of your choice.
  
  ### MacOS support will come later bc apple sucks >:[
  ![NOTE](https://img.shields.io/badge/NOTE:-orange) *I do not own any apple devices nor plan to, so mac support may take a while*
