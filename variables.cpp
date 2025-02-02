#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>

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

    for (int i = 0;i<pow(N,D); i++){
        setSpin(state, i, 1);
    }

    return state;
}

// Initializes state in hot system
vector<char> initHot(int D, int N, int S){
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

// appends new Column of data to csv file
void appendColumn(const string& filename, const string& new_column){
    ifstream infile(filename);
    ifstream infile2(new_column);
    ofstream temp_file("temp.csv");
    string line, line2;

    size_t i = 0; // Index for the new column values
    while (getline(infile2, line2)) {
        getline(infile, line);
        //cout << i << line << line2 << endl;
        // Append the new value to each line
        temp_file << line << line2 << ", ";

        temp_file << endl;
        i++;
    }

    infile.close();
    infile2.close();
    temp_file.close();

    // Replace the original file with the updated one
    remove(filename.c_str());
    rename("temp.csv", filename.c_str());
}

// Function to precompute the nearest neighbors for each point on the grid
vector<vector<int>> precomputeNeighbors(int D, int N) {
    vector<vector<int>> neighbors(pow(N, D), vector<int>(2 * D));

    for (int i = 0; i < pow(N, D); ++i) {
        vector<int> coordinates = indexToPosition(i, D, N);

        for (int d = 0; d < D; ++d) {
            vector<int> neighbor = coordinates;
            neighbor[d] = periodicBoundary(coordinates[d] + 1, N);
            neighbors[i][2 * d] = positionToIndex(neighbor, D, N);

            neighbor[d] = periodicBoundary(coordinates[d] - 1, N);
            neighbors[i][2 * d + 1] = positionToIndex(neighbor, D, N);
        }
    }

    return neighbors;
}

// Calculates the Hamiltonian with periodic boundary conditions and external field
double calculateHamiltonian(const vector<char> &state, const vector<vector<int>> &neighbors, int D, int N, double Beta, double B) {
    double H = 0.0;
    double h;

    for (int i = 0; i < pow(N, D); ++i) {
        for (int d = 0; d < D; ++d) {
            H -= Beta * getSpin(state, i) * getSpin(state, neighbors[i][2*d]);
        }
        H -= B * getSpin(state, i);
    }

    h = H / pow(N, D);
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
void metropolisStep(vector<char> &state, const vector<vector<int>> &neighbors, int D, int N, double Beta, double B, int S){
    static default_random_engine generator(S);
    uniform_real_distribution<double> dis(0,1);
    int count = 0;

    int totalSpins = pow(N, D);
    for (int i = 0; i < totalSpins; i++){
        double deltaH = 2*B*getSpin(state, i); 

        for (int d = 0; d<D; d++){
            deltaH += 2*Beta*getSpin(state, i)*(getSpin(state, neighbors[i][2*d]) + getSpin(state, neighbors[i][2*d+1]));
        }

        double r = dis(generator);

        if (exp(-deltaH) > r){
            flipSpin(state, i);
        }
    }
    
}

// Generates data for history plot in .csv file
void generateHistory(vector<char> &state, const vector<vector<int>> &neighbors, int D, int N, double Beta, double B, int M, int S, bool append = 0){
    ofstream outfile("Results/MagnetizationHistory.csv");
    ofstream outfile2("Results/EnergyHistory.csv");

    if (!outfile || !outfile2) {
        cerr << "File could not be opened!" << endl;
    }

    outfile << calculateMagnetization(state, D, N) << endl;

    for (int i = 0; i < M; i++){
        //cout << i << endl;
        metropolisStep(state, neighbors, D, N, Beta, B,  S);
        outfile << calculateMagnetization(state, D, N) << endl;
        outfile2 << calculateHamiltonian(state, neighbors, D, N, Beta, B) << endl;
    }

    outfile.close();
    outfile2.close();
    if (append){
        appendColumn("Results/MagnetizationReplica.csv", "Results/MagnetizationHistory.csv");
        appendColumn("Results/EnergyReplica.csv", "Results/EnergyHistory.csv");
    }
    
}



int main(){
    int D, N, S, M;
    double Beta, B;

    cout << "Enter the number of dimensions (D): ";
    cin >> D;

    cout << "Enter the number of points per dimension (N): ";
    cin >> N;

    cout << "Enter the length of the Markov-Chain (M): ";
    cin >> M;

    cout << "Enter the value for the coupling (Beta = J/kT): ";
    cin >> Beta;

    cout << "Enter the value for the magnetic field (B = b/kT): ";
    cin >> B;

    vector<vector<int>> neighbors = precomputeNeighbors(D, N);

    for (int i = 0; i<500; i++){
        cout << i << endl;
        vector<char> state = initHot(D, N, i);
        generateHistory(state, neighbors, D, N, Beta, B, M, i, 1);
    }
    

    return 0;
}