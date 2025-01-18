#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <curand_kernel.h>
#include <iostream>
#include <chrono>
#include <random>
#include <stdio.h>


using namespace std;
using namespace chrono;

// Global parameters
const int GRID_SIZE = 12; // Size of the simulation grid (GRID_SIZE x GRID_SIZE) max 32
const int INFECTION_RADIUS = 1; // Radius for infection spread
const float INFECTION_PROBABILITY = 0.3f; // Probability of infection
const int RECOVERY_TIME = 10; // Number of days for recovery
const int INITIAL_INFECTED = 4; // Initial number of infected cells
const int SIMULATION_DAYS = 20; // Total number of simulation days
const int NUM_STREAMS = 4; // Number of CUDA streams

enum State { SUSCEPTIBLE, INFECTED, RECOVERED };

// Define the Cell structure with aligned memory for coalesced access
struct Cell {
    int daysInfected; // Number of days a cell has been infected
    State state; // Current state of the cell (SUSCEPTIBLE, INFECTED, RECOVERED)
} __align__(8);

// Kernel to initialize random states for each thread
__global__ void initializeRandomStates(curandState* states, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < GRID_SIZE * GRID_SIZE && j < GRID_SIZE) {
        int idx = j * GRID_SIZE + i + offset;
        curand_init(1234, idx, 0, &states[idx]); // Initialize the random state for each thread
    }
}

// Kernel to simulate a single day of infection spread
__global__ void simulateDayKernel(Cell* grid, Cell* nextGrid, curandState* states, int offset) {
    // Shared memory for faster neighbor access
    extern __shared__ Cell sharedGrid[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + offset / GRID_SIZE;
    // Calculate local index in shared memory
    int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    if (localIdx == 0) {
        int start = offset > 0 ? offset - GRID_SIZE : 0;
        int finish = offset + (GRID_SIZE * GRID_SIZE) / NUM_STREAMS == GRID_SIZE * GRID_SIZE ? GRID_SIZE * GRID_SIZE : offset + (GRID_SIZE * GRID_SIZE) / NUM_STREAMS + GRID_SIZE;
        for (int lclidx = start; lclidx < finish; lclidx++) {
            sharedGrid[lclidx] = grid[lclidx];
        }
    }
    __syncthreads(); // Ensure all threads have loaded data into shared memory

    if (i < GRID_SIZE && j < GRID_SIZE)
    {
        int idx = j * GRID_SIZE + i;
        curandState localState = states[idx]; // Load the thread's random state

        Cell cell = sharedGrid[idx]; // Get the current cell's state
        Cell nextCell = cell; // Initialize the next state for the cell

        if (cell.state == SUSCEPTIBLE) {
            bool infected = false;

            // Check neighbors within the infection radius
            for (int di = -INFECTION_RADIUS; di <= INFECTION_RADIUS; ++di) {
                for (int dj = -INFECTION_RADIUS; dj <= INFECTION_RADIUS; ++dj) {
                    int ni = i + di;
                    int nj = j + dj;

                    // Ensure neighbor is within bounds of the block
                    if (ni >= 0 && ni < GRID_SIZE && nj >= 0 && nj < GRID_SIZE) {
                        if (sharedGrid[nj * GRID_SIZE + ni].state == INFECTED) {
                            // Check infection probability
                            if (curand_uniform(&localState) < 0.3f) {
                                infected = true;
                                break;
                            }
                        }
                    }
                }
                if (infected) break;
            }

            if (infected) {
                nextCell.state = INFECTED; // Update state to INFECTED
                nextCell.daysInfected = 0; // Reset infection counter
            }
        }
        else if (cell.state == INFECTED) {
            nextCell.daysInfected++; // Increment infection duration
            if (nextCell.daysInfected >= RECOVERY_TIME) {
                nextCell.state = RECOVERED; // Transition to RECOVERED state
            }
        }

        // Save the updated state to the next grid
        nextGrid[idx] = nextCell;
        states[idx] = localState; // Save the updated random state
    }
}

// Function to initialize the grid with initial states
void initializeGrid(Cell* grid) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, GRID_SIZE - 1);

    // Set all cells to SUSCEPTIBLE state
    for (int i = 0; i < GRID_SIZE * GRID_SIZE; ++i) {
        grid[i] = { 0, SUSCEPTIBLE };
    }

    // Infect a few random cells initially
    for (int i = 0; i < INITIAL_INFECTED; ++i) {
        int x = dis(gen);
        int y = dis(gen);
        grid[y * GRID_SIZE + x].state = INFECTED;
    }
}

// Function to display the grid states
void displayGrid(Cell* grid) {
    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            char c = '.';
            if (grid[i * GRID_SIZE + j].state == SUSCEPTIBLE) c = 'S';
            else if (grid[i * GRID_SIZE + j].state == INFECTED) c = 'I';
            else if (grid[i * GRID_SIZE + j].state == RECOVERED) c = 'R';

            cout << c << grid[i * GRID_SIZE + j].daysInfected << " ";
        }
        cout << "\n";
    }
    cout << "-------------------------------\n";
}

// Main function to simulate disease spread using GPU with streams
void simulateDiseaseSpreadingUnified(Cell* hostGrid) {
    size_t size = GRID_SIZE * GRID_SIZE * sizeof(Cell);
    size_t stateSize = GRID_SIZE * GRID_SIZE * sizeof(curandState);

    // Events creation
    cudaEvent_t startSim, stopSim;
    cudaEventCreate(&startSim);
    cudaEventCreate(&stopSim);

    Cell* grid; // Unified memory for the grid
    curandState* dStates;
    cudaMallocManaged(&grid, size); // Allocate Unified Memory
    cudaMalloc(&dStates, stateSize);

    const int NUM_STREAMS = 4; // Number of CUDA streams
    cudaStream_t streams[NUM_STREAMS];

    // Create CUDA streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threadsPerBlock(GRID_SIZE, GRID_SIZE / NUM_STREAMS);
    dim3 numBlocks(1, 1);

    // Initialize the grid and random states
    memcpy(grid, hostGrid, size); // Copy initial data to unified memory

    // Divide initialization across streams
    for (int streamId = 0; streamId < NUM_STREAMS; ++streamId) {
        int offset = streamId * ((GRID_SIZE * GRID_SIZE) / NUM_STREAMS);
        initializeRandomStates << <numBlocks, threadsPerBlock, size, streams[streamId] >> > (dStates, offset);
    }


    // Prefetch the data to the GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(grid, size, device, NULL);

    // Synchronize after initialization
    cudaDeviceSynchronize();

    displayGrid(grid);

    for (int day = 0; day < SIMULATION_DAYS; ++day) {
        // Divide simulation into regions processed by separate streams
        cudaEventRecord(startSim);
        for (int streamId = 0; streamId < NUM_STREAMS; ++streamId) {
            int offset = streamId * ((GRID_SIZE * GRID_SIZE) / NUM_STREAMS);
            simulateDayKernel << <numBlocks, threadsPerBlock, GRID_SIZE* GRID_SIZE * sizeof(Cell), streams[streamId] >> > (grid, grid, dStates, offset);
        }
        cudaEventRecord(stopSim);

        // Synchronize all streams to ensure computation is complete
        cudaDeviceSynchronize();

        // Count kernel time
        float timeSim;
        cudaEventElapsedTime(&timeSim, startSim, stopSim);

        // Display the grid (host and device share the same unified memory)
        cout << "Day " << day + 1 << " GPU Execution Time: " << timeSim << ":\n";
        displayGrid(grid);
    }

    // Free Unified Memory and other resources
    cudaFree(grid);
    cudaFree(dStates);

    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    // Destroy events
    cudaEventDestroy(startSim);
    cudaEventDestroy(stopSim);
}


int main() {
    Cell* grid = new Cell[GRID_SIZE * GRID_SIZE]; // Allocate host grid
    initializeGrid(grid); // Initialize grid with initial states

    auto start = high_resolution_clock::now(); // Start timing
    simulateDiseaseSpreadingUnified(grid); // Run the simulation on GPU
    auto end = high_resolution_clock::now(); // End timing

    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Total execution time: " << duration.count() << " ms" << endl;

    delete[] grid; // Free host memory
    return 0;
}
