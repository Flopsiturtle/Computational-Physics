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
void metropolisStep(vector<char> &state, const vector<vector<int>> &neighbors, int D, int N, double Beta, double B, default_random_engine &generator){
    
    uniform_real_distribution<double> dis(0,1);
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
void generateHistory(vector<char> &state, const vector<vector<int>> &neighbors, int D, int N, double Beta, double B, int M, int S, bool append, bool evol, const string& filenameM, const string& filenameE){
    ofstream outfile("Results/MagnetizationHistory.csv");
    ofstream outfile2("Results/EnergyHistory.csv");
    ofstream outfileState("stateEvolution.csv");

    if (!outfile || !outfile2 || !outfileState) {
        cerr << "File could not be opened!" << endl;
    }

    default_random_engine generator(S);
    outfile << calculateMagnetization(state, D, N) << endl;
    outfile2 << calculateHamiltonian(state, neighbors, D, N, Beta, B) << endl;

    for (int i = 0; i < M; i++){
        metropolisStep(state, neighbors, D, N, Beta, B, generator);
        outfile << calculateMagnetization(state, D, N) << endl;
        outfile2 << calculateHamiltonian(state, neighbors, D, N, Beta, B) << endl;

        if (evol){
            for (int j = 0; j < pow(N, D); j++) {
            outfileState << static_cast<int>(getSpin(state, j)) << " ";
            }
            outfileState << endl;
        }
        
    }

    outfile.close();
    outfile2.close();
    outfileState.close();
    if (append){
        appendColumn(filenameM, "Results/MagnetizationHistory.csv");
        appendColumn(filenameE, "Results/EnergyHistory.csv");
    }
    
}



int main(){
    int D, N, M;
    double B;

    cout << "Enter the number of dimensions (D): ";
    cin >> D;

    cout << "Enter the number of points per dimension (N): ";
    cin >> N;

    cout << "Enter the length of the Markov-Chain (M): ";
    cin >> M;

    cout << "Enter the value for the magnetic field (B = b/kT): ";
    cin >> B;


    vector<vector<int>> neighbors = precomputeNeighbors(D, N);

    //vector<double> betaValues = { 1/0.2, 1/0.6, 1.0, 1/1.4, 1/1.8, 1/2.2, 1/2.6, 1/3.0, 1/3.4, 1/3.8, 1/4.2, 1/4.6, 1/5.0, 1/5.4, 1/5.8, 1/6.2, 1/6.6, 1/7.0, 1/7.4, 1/7.8, 1/8.2, 1/8.6, 1/9.0, 1/9.4, 1/9.8};
    //vector<double> betaValues = {1/2.0, 1/2.3, 1/2.4, 1/2.5, 1/2.8};
    vector<double> betaValues = {1/2.0, 1/2.2};
    //vector<char> state = initHot(D, N, 19);
    //generateHistory(state, neighbors, D, N, 0.43, B, M, 19, 0, 1, "Results/MagnetizationHistory", "Results/EnergyHistory");
    
    for (double beta : betaValues) {
        string filenameM = "Results3/MagB" + to_string(B) + "Beta" + to_string(beta) + ".csv";
        string filenameE = "Results3/EnB" + to_string(B) + "Beta" + to_string(beta) + ".csv";

        for (int i = 0; i < 500; i++) {
            cout << "Beta: " << beta << ", Seed: " << i << endl;
            vector<char> state = initHot(D, N, i);
            generateHistory(state, neighbors, D, N, beta, B, M, i, 1, 0, filenameM, filenameE);
        }
    }
    

    return 0;
}


/// Plan for now: for specific values of Beta and B, generate 500 replicas of the system and calculate the magnetization and energy for each replica.
/// How do I save the data efficiently?