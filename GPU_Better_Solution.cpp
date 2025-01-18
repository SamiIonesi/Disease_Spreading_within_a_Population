#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>
#include <chrono>
#include <random>
#include <stdio.h>

using namespace std;
using namespace chrono;

// Parametri globali
const int GRID_SIZE = 12;
const int INFECTION_RADIUS = 1;
const float INFECTION_PROBABILITY = 0.3f;
const int RECOVERY_TIME = 10;
const int INITIAL_INFECTED = 4;
const int SIMULATION_DAYS = 20;
const int NUM_STREAMS = 4; // Number of CUDA streams

enum State { SUSCEPTIBLE, INFECTED, RECOVERED };

struct Cell {
    State state;
    int daysInfected;
};

// ???????  probabilitatea de infectie
// ce face mai exact pre?
// Deci aceasta functie genereaza pentru fiecare element din lista o prob. de infectie
__global__ void initializeRandomStates(curandState* states, int pre)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + pre % GRID_SIZE;
    int j = blockIdx.y * blockDim.y + threadIdx.y + pre / GRID_SIZE;

    if (i < GRID_SIZE && j < GRID_SIZE) {
        int idx = j * GRID_SIZE + i;
        curand_init(1234, idx, 0, &states[idx]);
    }
}

__global__ void simulateDayKernel(Cell* grid, Cell* nextGrid, curandState* states, int pre) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + pre % GRID_SIZE;
    int j = blockIdx.y * blockDim.y + threadIdx.y + pre / GRID_SIZE;
    int i_next = blockIdx.x * blockDim.x + threadIdx.x;
    int j_next = blockIdx.y * blockDim.y + threadIdx.y;

    // Aceasta asigură că, dacă o celulă nu este procesată (de exemplu, din cauza condițiilor), starea sa rămâne neschimbată.
    nextGrid[j_next * GRID_SIZE + i_next].state = grid[j * GRID_SIZE + i].state; 

    if (i < GRID_SIZE && j < GRID_SIZE) 
    {
        curandState localState = states[j * GRID_SIZE + i];

        if (grid[j * GRID_SIZE + i].state == SUSCEPTIBLE) 
        {
            bool infected = false;

            for (int di = -INFECTION_RADIUS; di <= INFECTION_RADIUS; ++di) 
            {
                for (int dj = -INFECTION_RADIUS; dj <= INFECTION_RADIUS; ++dj) 
                {
                    int ni = i + di;
                    int nj = j + dj;

                    if (ni >= 0 && ni < GRID_SIZE && nj >= 0 && nj < GRID_SIZE) 
                    {
                        if (grid[nj * GRID_SIZE + ni].state == INFECTED) 
                        {
                            if (curand_uniform(&localState) < 0.3f) 
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
                nextGrid[j_next * GRID_SIZE + i_next].state = INFECTED;
                nextGrid[j_next * GRID_SIZE + i_next].daysInfected = 0;
            }
        }
        else if (grid[j * GRID_SIZE + i].state == INFECTED) 
        {
            nextGrid[j_next * GRID_SIZE + i_next].daysInfected++;

            if (nextGrid[j_next * GRID_SIZE + i_next].daysInfected >= RECOVERY_TIME) 
            {
                nextGrid[j_next * GRID_SIZE + i_next].state = RECOVERED;
            }
        }

        states[j * GRID_SIZE + i] = localState;
    }
}

void initializeGrid(Cell* grid) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, GRID_SIZE - 1);

    for (int i = 0; i < GRID_SIZE; ++i) 
    {
        for (int j = 0; j < GRID_SIZE; ++j) 
        {
            grid[i * GRID_SIZE + j] = { SUSCEPTIBLE, 0 };
        }
    }
x`
    for (int i = 0; i < INITIAL_INFECTED; ++i) 
    {
        int x = dis(gen);
        int y = dis(gen);
        grid[x * GRID_SIZE + y].state = INFECTED;
    }
}

void displayGrid(const Cell* grid) 
{
    for (int i = 0; i < GRID_SIZE; ++i)
    {
        for (int j = 0; j < GRID_SIZE; ++j)
        {
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

void simulateDiseaseSpreadingGPU(Cell* hostGrid) 
{
    size_t size = GRID_SIZE * GRID_SIZE * sizeof(Cell);
    size_t stateSize = GRID_SIZE * GRID_SIZE * sizeof(curandState);

    Cell* dGrid, * dNextGrid;
    curandState* dStates;

    cudaMalloc(&dGrid, size);
    cudaMalloc(&dNextGrid, size);
    cudaMalloc(&dStates, stateSize);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) 
    {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threadsPerBlock(GRID_SIZE, GRID_SIZE / NUM_STREAMS);
    dim3 numBlocks(1, 1);

    cudaEvent_t startH2D, stopH2D, startSim, stopSim, startD2H, stopD2H;
    cudaEventCreate(&startH2D);
    cudaEventCreate(&stopH2D);
    cudaEventCreate(&startSim);
    cudaEventCreate(&stopSim);
    cudaEventCreate(&startD2H);
    cudaEventCreate(&stopD2H);

    cudaEventRecord(startH2D);

    cudaMemcpy(dGrid, hostGrid, size, cudaMemcpyHostToDevice);

    cudaEventRecord(stopH2D);

    for (int streamId = 0; streamId < NUM_STREAMS; ++streamId) 
    {
        initializeRandomStates << <numBlocks, threadsPerBlock, 0, streams[streamId] >> > (dStates, streamId * ((GRID_SIZE * GRID_SIZE) / NUM_STREAMS));
    }

    for (int day = 0; day < SIMULATION_DAYS; ++day) 
    {

        cudaDeviceSynchronize();

        displayGrid(hostGrid);

        cudaEventRecord(startSim);

        for (int streamId = 0; streamId < NUM_STREAMS; ++streamId) 
        {
            simulateDayKernel << < numBlocks, threadsPerBlock, 0, streams[streamId] >> > (dGrid, dNextGrid + streamId * ((GRID_SIZE * GRID_SIZE) / NUM_STREAMS), dStates, streamId * ((GRID_SIZE * GRID_SIZE) / NUM_STREAMS));
        }

        cudaEventRecord(stopSim);

        cudaEventRecord(startD2H);
        for (int streamId = 0; streamId < NUM_STREAMS; ++streamId) 
        {
            cudaMemcpyAsync(hostGrid + streamId * ((GRID_SIZE * GRID_SIZE) / NUM_STREAMS), dNextGrid + streamId * ((GRID_SIZE * GRID_SIZE) / NUM_STREAMS), size / NUM_STREAMS, cudaMemcpyDeviceToHost, streams[streamId]);
        }
        cudaEventRecord(stopD2H);

        for (int streamId = 0; streamId < NUM_STREAMS; ++streamId) 
        {
            cudaMemcpyAsync(dGrid + streamId * ((GRID_SIZE * GRID_SIZE) / NUM_STREAMS), dNextGrid + streamId * ((GRID_SIZE * GRID_SIZE) / NUM_STREAMS), size / NUM_STREAMS, cudaMemcpyDeviceToDevice, streams[streamId]);
        }


        float timeH2D, timeSim, timeD2H;
        cudaEventElapsedTime(&timeH2D, startH2D, stopH2D);
        cudaEventElapsedTime(&timeSim, startSim, stopSim);
        cudaEventElapsedTime(&timeD2H, startD2H, stopD2H);

        cout << "Day " << day + 1 << " GPU Execution Time: " << timeSim << " ms, Transfer H2D: " << timeH2D << " ms, Transfer D2H: " << timeD2H << " ms\n";
    }

    cudaEventRecord(startD2H);
    for (int streamId = 0; streamId < NUM_STREAMS; ++streamId) 
    {
        cudaMemcpyAsync(hostGrid + streamId * ((GRID_SIZE * GRID_SIZE) / NUM_STREAMS), dGrid + streamId * ((GRID_SIZE * GRID_SIZE) / NUM_STREAMS), size / NUM_STREAMS, cudaMemcpyDeviceToHost, streams[streamId]);
    }
    cudaEventRecord(stopD2H);

    cudaDeviceSynchronize();

    cudaMemcpy(hostGrid, dGrid, size, cudaMemcpyDeviceToHost);
    cudaFree(dGrid);
    cudaFree(dNextGrid);
    cudaFree(dStates);

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    cudaEventDestroy(startH2D);
    cudaEventDestroy(stopH2D);
    cudaEventDestroy(startSim);
    cudaEventDestroy(stopSim);
    cudaEventDestroy(startD2H);
    cudaEventDestroy(stopD2H);
}

int main() {
    Cell* grid = new Cell[GRID_SIZE * GRID_SIZE];
    initializeGrid(grid);

    auto start = high_resolution_clock::now();
    simulateDiseaseSpreadingGPU(grid);
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Total execution time: " << duration.count() << " ms" << endl;

    delete[] grid;
    return 0;
}
