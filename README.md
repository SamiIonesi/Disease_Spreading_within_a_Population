# Disease_Spreading_within_a_Population
This repository describe the simulation of spreading a disease within a population with the help of CPU in comparison with GPU.

## 1. The story
### 1.1. General description
We all know that we live in a world filled with many viruses and diseases, which can easily make us sick, and in some cases, even lead to death. We have already experienced the COVID-19 pandemic and witnessed the immense damage an unexpected illness can cause. For this reason, this project is designed to simulate the spread of a disease within a population, utilizing GPUs to accelerate the process. <br>

Each individual in the population is represented as a point on a two-dimensional grid, and their states evolve over time according to a simplified epidemiological model, such as the SIR (Susceptible, Infected, Recovered) model. <br>

This type of simulation is used to understand how diseases spread, test control strategies (e.g., vaccination, isolation), and analyze the effects of factors such as population density and contact rate. <br>

Simulations involving large populations (tens of thousands or even millions of individuals) require a high computational workload, making them well-suited for acceleration through parallel computing. <br>

### 1.2. Epidemological model

The SIR model is used, which assumes that individuals can exist in three distinct states:
- S (Susceptible): Healthy individuals who can be infected.
- I (Infected): Individuals who are sick and can infect others.
- R (Recovered): Individuals who were infected and can no longer transmit the disease (either immune or deceased).
  
Transition rules:
- A susceptible individual can become infected if they are near an infected individual, with a certain probability. <br>
- An infected individual transitions to the recovered state after an average infection duration. <br>

### 1.3. Algorithmic Components Requiring Parallel Computation

In the classical SIR model, there are several components that can benefit from parallel computation:

#### **1.3.1. Iterating through each individual in the population**

  Each individual has an initial state and a potential transition (e.g., from susceptible to infected, or from infected to recovered). The complexity of this process  is O(n x k), where:
- **n** is the number of individuals
- **k** is the average number of neighbors analyzed for each individual
  
#### **1.3.2. Analyzing the state of each individual**

  Each individual must evaluate the states of its neighbors to determine whether it will become infected. The following calculations are necessary:
- compute the probability of infection
- verify the grid boundaries and identify valid neighbors
  
#### **1.3.3. Analyzing statistical data**

  Calculate the proportions of infected, recovered, or susceptible individuals to generate statistics throughout the simulation.

### 1.4. Problem complexity

![Desese_Table](https://github.com/user-attachments/assets/a6d5562e-c769-4789-952a-1046db17104d)

The population is represented as a two-dimensional grid, forming an **n x n** matrix. <br>

Thus, there are **n^2** individuals in the population. <br>

For each step in the simulation, the complexity is **O(n^2 x k)**, where **k** is the number of neighbors analyzed per individual.

Example: 
- for *n = 5 & k = 9* we'll have **O(5^2 * 9) = 225**
- but for *n = 1.000 & k = 9* we'll have **O(1.000^2 * 9) = 9.000.000**
- and for *n = 1.000.000 & k = 9* we'll have **O(1.000.000^2 * 9) = 9.000.000.000.000**

### 1.5. Justification for Exploiting Parallelism

Parallel computing is justified due to the characteristics of the problem:

#### **1.5.1. Cell independence**

  Each cell (individual) can be evaluated independently within a time step. This makes the problem ideal for parallel computing, especially on GPUs.
  
#### **1.5.2. Large grid operation**

  GPUs are optimized for repetitive computations on structured data (e.g. 2D/3D grids).
  
#### **1.5.3. Parallel access to neighbors**

  Evaluation of neighbors for each cell can be distributed across multiple threads.
  
#### **1.5.4. Massive acceleration of execution time**

  On a CPU, a simulation for a million cells can take minutes or even hours, but on a well-optimized GPU, this time can be reduced to a few seconds.

### 1.6. Examples of Existing Products/Applications

#### 1.6.1. [EpiSimdemics](https://ieeexplore.ieee.org/document/5214892)

An efficient algorithm for simulating the spread of infectious disease over large realistic social networks

#### 1.6.2. [COVIDSim](https://github.com/mrc-ide/covid-sim)

A simulator for modeling the spread of COVID-19. It includes an advanced epidemiological model implemented in parallel to study lockdown and vaccination strategies.

#### 1.6.3. [AnyLogic](https://www.anylogic.com/)

A commercial simulation platform that integrates agent-based models. It uses parallel computing to simulate large population dynamics and disease spread.

## 2. Methods to solve the problem

### [2.1 CPU Naive Solution](https://github.com/SamiIonesi/Disease_Spreading_within_a_Population/blob/main/CPU_Naive_Solution.cpp)

#### 2.1.1.	How It Works:
•	The CPU solution uses a two-dimensional grid to represent the population. <br>
•	For each time step, it evaluates the state of each individual and their neighbors to determine state transitions (Susceptible → Infected, Infected → Recovered). <br>
•	It uses nested loops for grid traversal and probabilistic calculations for infection spread, which are computationally expensive for large grids. <br>
•	The simulation runs sequentially on the CPU, using a single thread. <br>

#### 2.1.2.	Key Features:
•	Iterates through all individuals in the grid. <br>
•	Checks all neighbors within the infection radius. <br>
•	Probabilities of infection are computed using random values. <br>
•	Uses nested loops for the time evolution of the grid. <br>

### 2.2. [GPU Naive Solution](https://github.com/SamiIonesi/Disease_Spreading_within_a_Population/blob/main/GPU_Naive_Solution.cpp)

#### 2.2.1	How It Works:
•	The GPU solution parallelizes the grid computation using CUDA kernels. <br>
•	Each thread evaluates the state of one grid cell, reducing the need for sequential iteration. <br>
•	It uses curand to generate random numbers for infection probability and state updates. <br>
•	The state grid is stored in global memory, and computations are parallelized across threads for cells in the grid. <br>
•	Memory transfer between host (CPU) and device (GPU) is managed before and after each simulation step. <br>

#### 2.2.2	Key Features: 
•	Each thread represents a grid cell. <br>
•	Exploits parallelism to evaluate the state transitions for all individuals simultaneously. <br>
•	Optimized for repetitive computations, leveraging CUDA’s memory and execution model. <br>
•	Synchronization is used to ensure thread safety for updates. <br>

### 2.3. GPU Improved Solution
  TO DO...

## 3.	Experemental results

### 3.1.	Details of hardware/software devices used in experiments - CPU, GPU, OS, CUDA version
•	**CPU**: 11th Gen Intel® Core™ i5-1135G7 @ 2.40GHz <br>
•	**GPU**: NVIDIA GeForce RTX 2060 SUPER <br>
•	**OS**: Windows 11 <br>
•	**CUDA Version**: 11.8 <br>
•	**Software**: NVIDIA CUDA Toolkit and the C++ compiler compatible with the CUDA runtime from Visual Studio. <br>

### 3.2.	Data sets used in experiments – describe what kind of data they contain, where/how they were obtained, size, etc.

#### 3.2.1.	Content:
•	The dataset consists of a two-dimensional array representing the population. Each element in the array is an Struct that contains: <br>
o	State (Susceptible, Infected, Recovered). <br>
o	daysInfected (number of days since infection). <br>
Example of the array of 10x10 with indivituals states and daysInfected: <br>

![image](https://github.com/user-attachments/assets/9f48a581-20d7-4da1-be78-ac79d410d465)

#### 3.2.2.	Generation: <br>
•	The array is initialized programmatically using random generators. A few cells are seeded as initially infected.

#### 3.2.3.	Size:
•	Small: 10×10 grid (100 individuals). <br>
•	Large: 1000×1000 grid (1,000,000 individuals). <br>

#### 3.2.4.	Source
•	Simulated within the code using random number generation.

#### 3.2.5.	Time of execution
The simulation was done for twenty days.

##### 3.2.5.1	CPU
•	Array size = 100, with display function
 ![image](https://github.com/user-attachments/assets/651c5aa4-e64f-4563-bc09-0c76397b1ac4)

•	Array size = 100, without display function
 ![image](https://github.com/user-attachments/assets/df8f1aaa-31ba-432e-8602-70e1497a067b)

•	Array size = 10.000, with display function
 ![image](https://github.com/user-attachments/assets/d13dc89b-393d-41df-a37e-ae2dc3a8797f)

•	Array size = 10.000, without display function
 ![image](https://github.com/user-attachments/assets/e0b047fe-0359-4533-8624-f5f37e0ed011)

•	Array size = 1.000.000, with display function
 ![image](https://github.com/user-attachments/assets/8687471a-913f-41f5-971d-8848fd63ba8e)

•	Array size = 1.000.000, without display function
 ![image](https://github.com/user-attachments/assets/4831c996-59f1-471c-8405-9856d7825986)

•	Array size = 100.000.000, with display function
 ![image](https://github.com/user-attachments/assets/cb5f6f68-4382-4bed-981f-b99cc0166a4f)

This means aprox. 12 minutes.
•	Array size = 100.000.000, without display function
 ![image](https://github.com/user-attachments/assets/dc82fbba-2e98-47d5-88c5-d66a420bf2d1)

This means aprox. 2 minutes.

##### 3.2.5.2	GPU
•	Array size = 100, with display function <br>
 ![image](https://github.com/user-attachments/assets/a7e5836b-9862-4596-9082-e85d4fcbfe67)

•	Array size = 100, without display function <br>
 ![image](https://github.com/user-attachments/assets/443805f0-b902-4fe8-bdd0-0b4ea9577de0)

•	Array size = 10.000, with display function <br>
 ![image](https://github.com/user-attachments/assets/270dd138-01f1-482e-b168-cf90553a2b5b)

•	Array size = 10.000, without display function <br>
 ![image](https://github.com/user-attachments/assets/38472f59-5889-4f82-a39f-18678cf602de)

•	Array size = 1.000.000, with display function <br>
More than 15 minutes. <br>

•	Array size = 1.000.000, without display function <br>
 ![image](https://github.com/user-attachments/assets/21ae10f5-3655-4517-a05b-4378dfba8528)

•	Array size = 100.000.000, with display function <br>
More than 25 minutes. <br>

•	Array size = 100.000.000, without display function <br>
 ![image](https://github.com/user-attachments/assets/5ad05215-df1a-497c-9de4-73f3ce8600af)


3.3.	Graph of scalability of the parallel solution(s) in relation to the size of the input data set, compared to the single-threaded CPU solution

| **CPU**                                                               |
| **Array Size**     | **With Display (ms)** | **Without Display (ms)** |
|--------------------|-----------------------|--------------------------|
| 100               | 0                     | 0                        |
| 10,000            | 42                    | 13                       |
| 1,000,000         | 5,310                 | 1,072                    |
| 100,000,000       | 705,695               | 125,923                  |



