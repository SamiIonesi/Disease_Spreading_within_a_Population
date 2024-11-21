# Disease_Spreading_within_a_Population
This repository describe the simulation of spreading a disease within a population with the help of CPU in comparison with GPU.

## The story
### General description
We all know that we live in a world filled with many viruses and diseases, which can easily make us sick, and in some cases, even lead to death. We have already experienced the COVID-19 pandemic and witnessed the immense damage an unexpected illness can cause. For this reason, this project is designed to simulate the spread of a disease within a population, utilizing GPUs to accelerate the process. <br>

Each individual in the population is represented as a point on a two-dimensional grid, and their states evolve over time according to a simplified epidemiological model, such as the SIR (Susceptible, Infected, Recovered) model. <br>

This type of simulation is used to understand how diseases spread, test control strategies (e.g., vaccination, isolation), and analyze the effects of factors such as population density and contact rate. <br>

### Epidemological model
The SIR model is used, which assumes that individuals can exist in three distinct states:
- S (Susceptible): Healthy individuals who can be infected.
- I (Infected): Individuals who are sick and can infect others.
- R (Recovered): Individuals who were infected and can no longer transmit the disease (either immune or deceased).
  
Transition rules:
- A susceptible individual can become infected if they are near an infected individual, with a certain probability. <br>
- An infected individual transitions to the recovered state after an average infection duration. <br>

## Methods to solve the problem

## Experimental results
