#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;
using namespace chrono;

// Global parameters
const int GRID_SIZE = 100;
const int INFECTION_RADIUS = 1;
const float INFECTION_PROBABILITY = 0.3f;
const int RECOVERY_TIME = 14;
const int INITIAL_INFECTED = 10;
const int SIMULATION_DAYS = 50;

enum State { SUSCEPTIBLE, INFECTED, RECOVERED };

struct Cell
{
    State state;
    int daysInfected;
};

void initializeGrid(vector<vector<Cell>>& grid)
{
    random_device random;       //this is basically a random number generator
    mt19937 gen(random());      //this one basically ensure that every number is distinct
    uniform_int_distribution<> dis(0, GRID_SIZE - 1);   //this one ensure that every number of the grid can have the same probability of generation


    //this one is used to initialize the grid with all individual having a SUSCEPTIBLE state
    for (int i = 0; i < GRID_SIZE; ++i) 
    {
        for (int j = 0; j < GRID_SIZE; ++j) 
        {
            grid[i][j] = {SUSCEPTIBLE, 0};
        }
    }

    //this one is used to initialize in a random order using the upper generator the infected individuals
    for (int i = 0; i < INITIAL_INFECTED; ++i) 
    {
        int x = dis(gen);
        int y = dis(gen);
        grid[x][y] = {INFECTED, 0};
    }

}

void displayGrid(const std::vector<std::vector<Cell>>& grid) {
    for (int i = 0; i < GRID_SIZE; ++i) 
    {
        for (int j = 0; j < GRID_SIZE; ++j) 
        {
            char c = '.';

            if (grid[i][j].state == SUSCEPTIBLE) c = 'S';
            else if (grid[i][j].state == INFECTED) c = 'I';
            else if (grid[i][j].state == RECOVERED) c = 'R';
            
            cout << c << " ";
        }
        
        cout << "\n";
    }
    
    cout << "-------------------------------\n";
}

void simulateDay(vector<vector<Cell>>& grid)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    auto next_grid = grid;

    for (int i = 0; i < GRID_SIZE; ++i) 
    {
        for (int j = 0; j < GRID_SIZE; ++j) 
        {
            if (grid[i][j].state == SUSCEPTIBLE) {

                bool infected = false;

                for (int di = -INFECTION_RADIUS; di <= INFECTION_RADIUS; ++di) 
                {
                    for (int dj = -INFECTION_RADIUS; dj <= INFECTION_RADIUS; ++dj) 
                    {
                        int ni = i + di;
                        int nj = j + dj;

                        if (ni >= 0 && ni < GRID_SIZE && nj >= 0 && nj < GRID_SIZE) 
                        {
                            if (grid[ni][nj].state == INFECTED) 
                            {
                                if (dis(gen) < INFECTION_PROBABILITY) //for every infected neighbor we verify if the probability is less tha 0.3
                                {
                                    infected = true;
                                    break;
                                }
                            }
                        }
                    }
                    
                    if (infected) break;
                }

                if (infected) 
                {
                    next_grid[i][j].state = INFECTED;
                    next_grid[i][j].daysInfected = 0;
                }

            } 
            
            else if (grid[i][j].state == INFECTED) 
            {
                next_grid[i][j].daysInfected++;

                if (next_grid[i][j].daysInfected >= RECOVERY_TIME) 
                {
                    next_grid[i][j].state = RECOVERED;
                }
            }

        }
    }

    grid = next_grid;
}

void countIndividuals(const vector<vector<Cell>>& grid, int& susceptible, int& infected, int& recovered) {
    susceptible = 0;
    infected = 0;
    recovered = 0;

    for (const auto& row : grid) {
        for (const auto& cell : row) {
            if (cell.state == SUSCEPTIBLE) susceptible++;
            else if (cell.state == INFECTED) infected++;
            else if (cell.state == RECOVERED) recovered++;
        }
    }
}

void simulateDiseaseSpreading(vector<vector<Cell>>& grid)
{
    initializeGrid(grid);

    for (int day = 0; day < SIMULATION_DAYS; ++day) 
    {
        int susceptible, infected, recovered;
        countIndividuals(grid, susceptible, infected, recovered);

        cout << "Day " << day + 1 << ":     "
        << "SUSCEPTIBLE = " << susceptible << "     "
        << "INFECTED = " << infected << "     "
        << "RECOVERED = " << recovered << "     "
        << endl;

        displayGrid(grid);
        simulateDay(grid);
    }
}

int main()
{
    vector<vector<Cell>> grid(GRID_SIZE, vector<Cell>(GRID_SIZE));

    //This is used to start measuring the time on CPU
    auto start_time = high_resolution_clock::now();

    simulateDiseaseSpreading(grid);

    //This is used to stop the measuring the time on CPU
    auto end_time = high_resolution_clock::now();

    //This one is used to calculate how mutch time CPU takes to perform the simulation
    auto simulationDuration = duration_cast<milliseconds>(end_time - start_time);

    //Display the time
    cout << "Total execution time for CPU is " << simulationDuration.count() << " miliseconds." << endl; 

    return 0;
}