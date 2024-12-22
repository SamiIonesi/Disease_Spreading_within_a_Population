#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>
#include <chrono>
#include <random>

using namespace std;
using namespace chrono;

// Parametri globali
const int GRID_SIZE = 10;
const int INFECTION_RADIUS = 1;
const float INFECTION_PROBABILITY = 0.3f;
const int RECOVERY_TIME = 10;
const int INITIAL_INFECTED = 4;
const int SIMULATION_DAYS = 20;

// Stările celulei
enum State { SUSCEPTIBLE, INFECTED, RECOVERED };

struct Cell {
    State state;
    int daysInfected;
};

// Kernel pentru inițializarea generatorului cuRAND
//Practic aceasta functie va grea un vector unidimesnional in care va salva o valoare
//aleatoare pentru fiecare element din grila, adica pentru fiecare thread

__global__ void initializeRandomStates(curandState* states, int gridSize)
{
    //pentru a determina threadul global corespunzator fiecarui celule (i, j)
    int i = blockIdx.x * blockDim.x + threadIdx.x; //Linia curenta
    int j = blockIdx.y * blockDim.y + threadIdx.y; //Coloana curenta

    if (i < gridSize && j < gridSize) //verificam daca suntem in interiorul grilei
    {
        int idx = i * gridSize + j;
        curand_init(1234, idx, 0, &states[idx]); // Initializam generatorul curand
    }
}

// __global__ spune ca aceasta functie va fi apelata de catre CPU dar fiecare thread va rula in paralel pe GPU
// Kernel pentru simularea unei zile
__global__ void simulateDayKernel(Cell* grid, Cell* nextGrid, curandState* states, int gridSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    nextGrid[i * gridSize + j] = grid[i * gridSize + j];

    if (i < gridSize && j < gridSize)
    {
        //se foloseste un generator local ca si copie a generatorului global
        //pentru a evita acesul la memoria globala
        curandState localState = states[i * gridSize + j]; // Generator local

        if (grid[i * gridSize + j].state == SUSCEPTIBLE)
        {
            bool infected = false;

            // Verificăm vecinii în raza de infecție
            for (int di = -INFECTION_RADIUS; di <= INFECTION_RADIUS; ++di) {
                for (int dj = -INFECTION_RADIUS; dj <= INFECTION_RADIUS; ++dj) {
                    int ni = i + di;
                    int nj = j + dj;

                    if (ni >= 0 && ni < gridSize && nj >= 0 && nj < gridSize)
                    {
                        if (grid[ni * gridSize + nj].state == INFECTED)
                        {
                            float prob = curand_uniform(&localState); // Generăm un număr aleator între [0, 1]
                            if (prob < 0.3f) //INFECTION_PROBABILITY
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
                nextGrid[i * gridSize + j].state = INFECTED;
                nextGrid[i * gridSize + j].daysInfected = 0;
            }
        }
        else if (grid[i * gridSize + j].state == INFECTED)
        {
            nextGrid[i * gridSize + j].daysInfected = nextGrid[i * gridSize + j].daysInfected + 1;

            if (nextGrid[i * gridSize + j].daysInfected >= RECOVERY_TIME)
            {
                nextGrid[i * gridSize + j].state = RECOVERED;
            }
        }

        states[i * gridSize + j] = localState; // Salvăm starea generatorului
    }
}

void initializeGrid(Cell* grid)
{
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

    for (int i = 0; i < INITIAL_INFECTED; ++i)
    {
        int x = dis(gen);
        int y = dis(gen);
        grid[x * GRID_SIZE + y].state = INFECTED;
    }
}

void countIndividuals(Cell* grid, int& susceptible, int& infected, int& recovered) {
    susceptible = 0;
    infected = 0;
    recovered = 0;

    for (int i = 0; i < GRID_SIZE; ++i)
    {
        for (int j = 0; j < GRID_SIZE; ++j)
        {
            if (grid[i * GRID_SIZE + j].state == SUSCEPTIBLE) susceptible++;
            else if (grid[i * GRID_SIZE + j].state == INFECTED) infected++;
            else if (grid[i * GRID_SIZE + j].state == RECOVERED) recovered++;
        }
    }
}

void displayGrid(Cell* hostGrid) {
    for (int i = 0; i < GRID_SIZE; ++i)
    {
        for (int j = 0; j < GRID_SIZE; ++j)
        {
            char c = '.';

            if (hostGrid[i * GRID_SIZE + j].state == SUSCEPTIBLE) c = 'S';
            else if (hostGrid[i * GRID_SIZE + j].state == INFECTED) c = 'I';
            else if (hostGrid[i * GRID_SIZE + j].state == RECOVERED) c = 'R';

            cout << c << hostGrid[i * GRID_SIZE + j].daysInfected << " ";
        }

        cout << "\n";
    }

    cout << "-------------------------------\n";
}

void simulateDiseaseSpreadingGPU(Cell* hostGrid)
{
    int susceptible, infected, recovered;
    size_t Size = GRID_SIZE * GRID_SIZE * sizeof(Cell); //dimensiunea in bytes a unei thread din grila
    size_t stateSize = GRID_SIZE * GRID_SIZE * sizeof(curandState); //dimensiunea vectorului e stari states

    //declaram pointari pentru grilele si vectorul in care se va face simularea pe GPU
    Cell *dGrid, *dNextGrid;
    curandState *dStates;

    // Timere CUDA
    // se creaza evenimente pentru timpul de simulare si cel de transfer
    cudaEvent_t startH2D, stopH2D, startSim, stopSim, startD2H, stopD2H;
    cudaEventCreate(&startH2D);
    cudaEventCreate(&stopH2D);
    cudaEventCreate(&startSim);
    cudaEventCreate(&stopSim);
    cudaEventCreate(&startD2H);
    cudaEventCreate(&stopD2H);

    // Alocare memorie pe GPU
    // aici se aloca memorie pentru pointeri
    cudaMalloc(&dGrid, Size);
    cudaMalloc(&dNextGrid, Size);


    //se aloca memorie pentru vectorul de stari
    cudaMalloc(&dStates, stateSize);

    // Transfer H2D
    cudaEventRecord(startH2D);

    cudaMemcpy(dGrid, hostGrid, Size, cudaMemcpyHostToDevice);

    cudaEventRecord(stopH2D);

    //aici are loc configurarea kernelului, se creaza atatea blocuri cat este nevoie pentru matrice
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((GRID_SIZE + 7) / 8, (GRID_SIZE + 7) / 8);

    // Inițializarea generatorului
    cudaEventRecord(startSim);
    //se creiaza vectorul de stari aleatoare
    initializeRandomStates << <numBlocks, threadsPerBlock >> > (dStates, GRID_SIZE);
    cudaDeviceSynchronize();

    // Rularea kernelului pentru simulare
    for (int day = 0; day < SIMULATION_DAYS; ++day) {
        simulateDayKernel << <numBlocks, threadsPerBlock >> > (dGrid, dNextGrid, dStates, GRID_SIZE);

        cudaDeviceSynchronize(); // se foloseste pentru a astepta terminarea fiecarui kernel

        swap(dGrid, dNextGrid);

        {
            cudaMemcpy(hostGrid, dGrid, Size, cudaMemcpyDeviceToHost);
        }

        countIndividuals(hostGrid, susceptible, infected, recovered);

        cout << "Day " << day + 1 << ":     "
            << "SUSCEPTIBLE = " << susceptible << "     "
            << "INFECTED = " << infected << "     "
            << "RECOVERED = " << recovered << "     "
            << endl;

        displayGrid(hostGrid);
    }

    cudaEventRecord(stopSim);

    // Transfer D2H
    cudaEventRecord(startD2H);
    cudaMemcpy(hostGrid, dGrid, Size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopD2H);

    // Afișarea timpilor
    cudaEventSynchronize(stopH2D);
    cudaEventSynchronize(stopSim);
    cudaEventSynchronize(stopD2H);

    float timeH2D, timeSim, timeD2H;
    cudaEventElapsedTime(&timeH2D, startH2D, stopH2D);
    cudaEventElapsedTime(&timeSim, startSim, stopSim);
    cudaEventElapsedTime(&timeD2H, startD2H, stopD2H);

    cout << "Transfer H2D Time: " << timeH2D << " ms\n";
    cout << "GPU Simulation Time: " << timeSim << " ms\n";
    cout << "Transfer D2H Time: " << timeD2H << " ms\n";


    // for (int i = 0; i < GRID_SIZE; ++i) {
    //     cudaFree(dGrid);
    //     cudaFree(dNextGrid);
    // }

    // Eliberarea memoriei
    cudaFree(dGrid);
    cudaFree(dNextGrid);
    cudaFree(dStates);

    //Distrugere evenimente
    cudaEventDestroy(startH2D);
    cudaEventDestroy(stopH2D);
    cudaEventDestroy(startSim);
    cudaEventDestroy(stopSim);
    cudaEventDestroy(startD2H);
    cudaEventDestroy(stopD2H);
}

int main()
{
    Cell* grid = new Cell [GRID_SIZE * GRID_SIZE]; //alocam memorie pentru grila

    initializeGrid(grid); //se initializeaza grila pe host

    simulateDiseaseSpreadingGPU(grid); //se face simularea pe GPU

    delete[] grid; //se dealoca memoria pentru grila

    return 0;
}