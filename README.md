# Disease_Spreading_within_a_Population
This repository describe the simulation of spreading a disease within a population with the help of CPU in comparison with GPU.

## 1. The story
### 1.1 General description
We all know that we live in a world filled with many viruses and diseases, which can easily make us sick, and in some cases, even lead to death. We have already experienced the COVID-19 pandemic and witnessed the immense damage an unexpected illness can cause. For this reason, this project is designed to simulate the spread of a disease within a population, utilizing GPUs to accelerate the process. <br>

Each individual in the population is represented as a point on a two-dimensional grid, and their states evolve over time according to a simplified epidemiological model, such as the SIR (Susceptible, Infected, Recovered) model. <br>

This type of simulation is used to understand how diseases spread, test control strategies (e.g., vaccination, isolation), and analyze the effects of factors such as population density and contact rate. <br>

Simulations involving large populations (tens of thousands or even millions of individuals) require a high computational workload, making them well-suited for acceleration through parallel computing. <br>

### 1.2 Epidemological model

The SIR model is used, which assumes that individuals can exist in three distinct states:
- S (Susceptible): Healthy individuals who can be infected.
- I (Infected): Individuals who are sick and can infect others.
- R (Recovered): Individuals who were infected and can no longer transmit the disease (either immune or deceased).
  
Transition rules:
- A susceptible individual can become infected if they are near an infected individual, with a certain probability. <br>
- An infected individual transitions to the recovered state after an average infection duration. <br>

### 1.3 Algorithmic Components Requiring Parallel Computation

In the classical SIR model, there are several components that can benefit from parallel computation:

#### **1.3.1 Iterating through each individual in the population**

  Each individual has an initial state and a potential transition (e.g., from susceptible to infected, or from infected to recovered). The complexity of this process  is O(n x k), where:
- **n** is the number of individuals
- **k** is the average number of neighbors analyzed for each individual
  
#### **1.3.2 Analyzing the state of each individual**

  Each individual must evaluate the states of its neighbors to determine whether it will become infected. The following calculations are necessary:
- compute the probability of infection
- verify the grid boundaries and identify valid neighbors
  
#### **1.3.3 Analyzing statistical data**

  Calculate the proportions of infected, recovered, or susceptible individuals to generate statistics throughout the simulation.

### 1.4 Problem complexity

![Desese_Table](https://github.com/user-attachments/assets/a6d5562e-c769-4789-952a-1046db17104d)

The population is represented as a two-dimensional grid, forming an **n x n** matrix. <br>

Thus, there are **n^2** individuals in the population. <br>

For each step in the simulation, the complexity is **O(n^2 x k)**, where **k** is the number of neighbors analyzed per individual.

Example: 
- for *n = 5 & k = 9* we'll have **O(5^2 * 9) = 225**
- but for *n = 1.000 & k = 9* we'll have **O(1.000^2 * 9) = 9.000.000**
- and for *n = 1.000.000 & k = 9* we'll have **O(1.000.000^2 * 9) = 9.000.000.000.000**

### 1.5 Justification for Exploiting Parallelism

Parallel computing is justified due to the characteristics of the problem:

#### **1.5.1 Cell independence**

  Each cell (individual) can be evaluated independently within a time step. This makes the problem ideal for parallel computing, especially on GPUs.
  
#### **1.5.2 Large grid operation**

  GPUs are optimized for repetitive computations on structured data (e.g. 2D/3D grids).
  
#### **1.5.3 Parallel access to neighbors**

  Evaluation of neighbors for each cell can be distributed across multiple threads.
  
#### **1.5.4 Massive acceleration of execution time**

  On a CPU, a simulation for a million cells can take minutes or even hours, but on a well-optimized GPU, this time can be reduced to a few seconds.

### 1.6 Examples of Existing Products/Applications

#### 1.6.1 [EpiSimdemics](https://ieeexplore.ieee.org/document/5214892)

An efficient algorithm for simulating the spread of infectious disease over large realistic social networks

#### 1.6.2 [COVIDSim](https://github.com/mrc-ide/covid-sim)

A simulator for modeling the spread of COVID-19. It includes an advanced epidemiological model implemented in parallel to study lockdown and vaccination strategies.

#### 1.6.3 [AnyLogic](https://www.anylogic.com/)

A commercial simulation platform that integrates agent-based models. It uses parallel computing to simulate large population dynamics and disease spread.

## 2. Methods to solve the problem

### [2.1 CPU Naive Solution](https://github.com/SamiIonesi/Disease_Spreading_within_a_Population/blob/main/CPU_Naive_Solution.cpp)

#### 2.1.1	How It Works:
•	The CPU solution uses a two-dimensional grid to represent the population.
•	For each time step, it evaluates the state of each individual and their neighbors to determine state transitions (Susceptible → Infected, Infected → Recovered).
•	It uses nested loops for grid traversal and probabilistic calculations for infection spread, which are computationally expensive for large grids.
•	The simulation runs sequentially on the CPU, using a single thread.

#### 2.1.2	Key Features:
•	Iterates through all individuals in the grid.
•	Checks all neighbors within the infection radius.
•	Probabilities of infection are computed using random values.
•	Uses nested loops for the time evolution of the grid.

### 2.2 [GPU Naive Solution]()

#### 2.2.1	How It Works:
•	The GPU solution parallelizes the grid computation using CUDA kernels.
•	Each thread evaluates the state of one grid cell, reducing the need for sequential iteration.
•	It uses curand to generate random numbers for infection probability and state updates.
•	The state grid is stored in global memory, and computations are parallelized across threads for cells in the grid.
•	Memory transfer between host (CPU) and device (GPU) is managed before and after each simulation step.
#### 2.2.2	Key Features:
•	Each thread represents a grid cell.
•	Exploits parallelism to evaluate the state transitions for all individuals simultaneously.
•	Optimized for repetitive computations, leveraging CUDA’s memory and execution model.
•	Synchronization is used to ensure thread safety for updates.

### 3. GPU Improved Solution

## Experimental results
