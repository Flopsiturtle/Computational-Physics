#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>

#include <random>

#include <cstdlib>
#include <ctime>

using namespace std;



// Helper to calculate periodic boundary conditions
int periodicBoundary(int index, int maxIndex){
    return (index + maxIndex) % maxIndex;
}

// Gets the nth spin from the state (1 for up, -1 for down)
int getSpin(const vector<char> &state, int n) {
    int byteIndex = n / 8;
    int bitIndex = n % 8;
    return (state[byteIndex] & (1 << bitIndex)) ? 1 : -1;
}

// Sets the nth spin in the state (1 for up, 0 for down)
void setSpin(vector<char> &state, int n, int spin) {
    int byteIndex = n / 8;
    int bitIndex = n % 8;

    if (spin == 1) {
        state[byteIndex] |= (1 << bitIndex);
    } else {
        state[byteIndex] &= ~(1 << bitIndex);
    }

}

// Flips the nth spin in the state
void flipSpin(vector<char> &state, int n) {
    int byteIndex = n / 8;
    int bitIndex = n % 8;
    state[byteIndex] ^= (1 << bitIndex);
}

// Initializes state in cold system
vector<char> initCold(int D, int N){
    vector<char> state((pow(N,D) + 7) / 8, 0);

    for (int i = 0; i<pow(N,D); i++){
        setSpin(state, i, 1);
    }

    return state;
}

// Initializes state in hot system
vector<int> initHot(int D, int N, int S){
    srand(S);
    vector<int> state(pow(N, D), 0);

    for (int i = 0; i<state.size();  i++){
        state[i] = rand() % 2 ? 1 : -1; // Random spin (+1 or -1)
    }

    return state;
}

// Initializes state in hot system
vector<char> initHot2(int D, int N, int S){
    vector<char> state((pow(N,D) + 7) / 8, 0);
    default_random_engine generator(S); 
    uniform_int_distribution<> dis(1, 2);

    for (int i = 0; i<pow(N, D);  i++){
        int rand = dis(generator);
        if (rand % 2){
            setSpin(state, i, 1);
        } else{
            setSpin(state, i, 0);
        }
    }

    return state;
}

// Convert index to position on the grid
vector<int> indexToPosition(int index, int D, int N) {
    vector<int> position(D, 0);

    for (int d = 0; d < D; ++d) {
        position[d] = periodicBoundary(index, N);
        index /= N;
    }

    if (index != 0){
        cout << "index not in range!" << endl;
    }

    return position;
}

// Convert position on the grid to index
int positionToIndex(const vector<int> &position, int D, int N) {
    int index = 0;

    for (int d = D - 1; d >= 0; --d) {
        index = index * N + position[d];
    }

    return index;
}

// Helper to print Grid in 1D
void printGrid1D(vector<char> grid, int N){
    cout << "[";

    for (int i = 0; i<N; i++){
        cout << (getSpin(grid, i) == 1 ? " 1" : "-1");
        }

    cout << " ]" << endl;
}
// Helper to print Grid in 2D
void printGrid2D(vector<char> grid, int N){
    cout << "Grid:" << endl;
    for (int j = 0; j<N; j++){
        cout << "[";
        for (int i = 0; i<N; i++){
            int index = positionToIndex({i, j}, 2, N);
            cout << (getSpin(grid, index) == 1 ? " 1" : "-1");
        }
        cout << " ]" << endl;
    }
}


// Calculates the Hamiltonian with periodic boundary conditions and external field
double calculateHamiltonian(const vector<char> &state, int D, int N, double Beta, double B) {
    double H = 0.0;
    double h;

    for (int i = 0; i < pow(N, D); ++i) {
        vector<int> coordinates = indexToPosition(i, D, N);

        for (int d=0; d<D; ++d) {
            vector<int> neighbor = coordinates;
            neighbor[d] = periodicBoundary(coordinates[d] + 1, N);

            int neighborIndex = positionToIndex(neighbor, D, N);
            H -= Beta * getSpin(state, i) * getSpin(state, neighborIndex);
        }
        H -= B * getSpin(state, i);
    }

    h = H/pow(N, D);
    return H;
}

// Calculates the magnetization for a given state
double calculateMagnetization(const vector<char> &state, int D, int N) {
    double m, M = 0.0;

    for (int i = 0; i<pow(N,D); i++){
        M += getSpin(state, i);
    }
    m = M/pow(N,D);
    return M;
}

// Calculates one step of the Metropolis algorithm
void metropolisStep(vector<char> &state, int D, int N, double Beta, double B, int S){
    static default_random_engine generator(S);
    uniform_real_distribution<double> dis(0,1);
    int count = 0;

    for (int i = 0; i<pow(N, D); i++){
        //vector<char> guess = state;
        //flipSpin(guess, i);
        double deltaH = 2*B*getSpin(state, i); 
        vector<int> coordinates = indexToPosition(i, D, N);

        for (int d = 0; d<D; d++){
            vector<int> neighbor = coordinates;
            neighbor[d] = periodicBoundary(coordinates[d] + 1, N);
            int index1 = positionToIndex(neighbor, D, N);
            neighbor[d] = periodicBoundary(coordinates[d] - 1, N);
            int index2 = positionToIndex(neighbor, D, N);
            deltaH += 2*Beta*getSpin(state, i)*(getSpin(state, index1) + getSpin(state, index2));
        }

        //double deltaHOld = calculateHamiltonian(guess, D, N, Beta, B) - calculateHamiltonian(state, D, N, Beta, B);
        double r = dis(generator);

        //if (deltaH != deltaHOld){
        //    cout << "diff: " << deltaH-deltaHOld << endl;
        //} 
        
        //if (i == pow(N, D)/2){
        //    cout << exp(-deltaH) << " > " <<  r << " ? " << endl;
        //}
        if (exp(-deltaH) > r){
            flipSpin(state, i);
        }
        //cout << "deltaH: " <<  deltaH << endl;
    }
    
}

// Generates data for history plot in .csv file
void generateHistory(vector<char> &state, int D, int N, double Beta, double B, int M, int S){
    ofstream outfile("Results/History.csv");

    if (!outfile) {
        cerr << "File could not be opened!" << endl;
    }

    outfile << 0 << "," << calculateMagnetization(state, D, N) << endl;

    for (int i = 0; i<M; i++){
        cout << i << endl;
        metropolisStep(state, D, N, Beta, B,  S);
        outfile << i+1 << "," << calculateMagnetization(state, D, N) << endl;
    }

    outfile.close();
}



int main(){
    int D, N, S;
    double Beta, B;

    cout << "Enter the number of dimensions (D): ";
    cin >> D;

    cout << "Enter the number of points per dimension (N): ";
    cin >> N;

    cout << "Enter the seed to use for random numbers (S): ";
    cin >> S;

    cout << "Enter the value for the coupling (Beta = J/kT): ";
    cin >> Beta;

    cout << "Enter the value for the magnetic field (B = b/kT): ";
    cin >> B;


    vector<char> state = initHot2(D, N, S);

    //metropolisStep(state, D, N, Beta, B, S);

    //printGrid2D(state, N);
    //cout << calculateHamiltonian(state, D, N, Beta, B) << endl;
    generateHistory(state, D, N, Beta, B, 1000, S);
    //printGrid2D(state, N);
    //metropolisStep(state, D, N, Beta, B, S);
    //printGrid2D(state, N);

    return 0;
}