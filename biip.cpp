#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <map>
#include <list>
#include <algorithm>
#include <omp.h>
#include <queue>          // std::queue
#include <utility>

extern "C" {
#include "gurobi_c.h"
}

using namespace std;

// For convenient trailing-return-types in C++11:
#define AUTO_RETURN(...) \
 noexcept(noexcept(__VA_ARGS__)) -> decltype(__VA_ARGS__) {return (__VA_ARGS__);}

template<typename T>
constexpr auto decayed_begin(T &&c)
AUTO_RETURN(std::begin(std::forward<T>(c)))

template<typename T>
constexpr auto decayed_end(T &&c)
AUTO_RETURN(std::end(std::forward<T>(c)))

template<typename T, std::size_t N>
constexpr auto decayed_begin(T(&c)[N])
AUTO_RETURN(reinterpret_cast<typename std::remove_all_extents<T>::type *>(c    ))

template<typename T, std::size_t N>
constexpr auto decayed_end(T(&c)[N])
AUTO_RETURN(reinterpret_cast<typename std::remove_all_extents<T>::type *>(c + N))

// Definitions
typedef struct IntPair {
    int x;
    int y;
} IntPair;

typedef struct Quad {
    int IL;
    int JL;
    int IR;
    int JR;
} Quad;

struct IntPairCompare {
    bool operator()(const IntPair &first, const IntPair &second) const {
        return first.x < second.x || (first.x == second.x && first.y < second.y);
    }
};

struct Coord {
    int x, y;

    Coord(int _x, int _y) {
        x = _x;
        y = _y;
    }
};

// Function Prototypes
void graphviz(const double *);

int solve_gurobi(int I, int J, int *rmask, int *lmask, int *enoarr, int &newN, int &newLN, int &newRN, int &newM);

void quad_search(int IL, int JL, int IR, int JR);

int shall_we_solve(int I, int J, int newLN, int newRN);

void update_coord_list(int I, int J);

void mask_nodes(int I, int J, int *rmask, int *lmask, int *enoarr, int &newN, int &newLN, int &newRN, int &newM);

void execute_tasks();


/// Variables initialized once and not changed
// Number of nodes, edges
int N, M;
// Number of left nodes, right nodes
int LN, RN;

map<IntPair, int, IntPairCompare> EDGE_MAP;
vector<vector<int> *> L_NODES;
vector<vector<int> *> R_NODES;
vector<string> L_LABELS;
vector<string> R_LABELS;
int *R_MASK;
int *L_MASK;
int *ENO_ARR;

/// Result variables
double MAX_OBJ_VAL = 0;
double *MAX_SOL;         // keeps the solution array when the obj val is found maximum
int maxLN, maxRN, maxN, maxM;
int resultI, resultJ;

/// Variables used in only gurobi solve
GRBenv *gurobi_env = nullptr;

/// Used in multiple functions
list<Coord> COORD_LIST;

/// OpenMP Variables
omp_lock_t coord_list_lock;
std::queue<Quad> task_queue;
int worker_count = 0;
omp_lock_t queue_lock;
omp_lock_t worker_count_lock;

// For calculation of seconds
double diff_clock(clock_t clock1, clock_t clock2) {
    double diff_ticks = (double) clock1 - (double) clock2;
    double diff_ms = (diff_ticks * 10) / CLOCKS_PER_SEC;
    return diff_ms;
}

long int returnMilliSeconds() {
    struct timeval tp{};
    gettimeofday(&tp, nullptr);
    long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
    return ms;
}

void print_list(double *list, int size) {
    cout << "[";
    for (int i = 0; i < size - 1; i++) {
        cout << list[i] << " ";
    }
    cout << list[size - 1] << "]" << endl;
}

int main(int argc, char *argv[]) {
    int THREAD_LIMIT = 4;
    string INPUT_FILE = "./data/all_gene_disease_associations-gurobi.txt";
    string OUTPUT_FILE = "./biip_parallel_output.txt";

    if (argc > 1) {
        THREAD_LIMIT = stoi(argv[1]);
    }

    if (argc > 2) {
        INPUT_FILE = argv[2];
    }

    if (argc > 3) {
        OUTPUT_FILE = argv[3];
    }

    printf("BIIP Sequential\n");
    printf("Thread Limit: %d\t Input file: %s\n", THREAD_LIMIT, INPUT_FILE.c_str());

    int i, j, eno, isol;
    long int timeBefore;
    long int timeAfter;

    /// Read the graph
    int l_count = 0; // Number of nodes in the left partition
    int r_count = 0; // Number of nodes in the right partition
    int m_count = 0; // Number of edges in the graph

    L_NODES.push_back(nullptr);
    R_NODES.push_back(nullptr);
    L_LABELS.emplace_back("");
    R_LABELS.emplace_back("");

    string firstNode, secondNode;
    int idx_firstNode, idx_secondNode;
    map<string, int>::iterator itm;     // Iterator
    vector<int> *adj_vector;            // Adjacency vector
    map<string, int> linkMap;

    fstream input;
    input.open(INPUT_FILE, ios::in);

    while (input >> firstNode) {
        input >> secondNode;
        itm = linkMap.find(firstNode);
        if (itm != linkMap.end()) {
            idx_firstNode = itm->second;
        } else {
            l_count++;
            linkMap[firstNode] = l_count;
            idx_firstNode = l_count;
            adj_vector = new vector<int>;
            L_NODES.push_back(adj_vector);
            L_LABELS.push_back(firstNode);
        }
        itm = linkMap.find(secondNode);
        if (itm != linkMap.end()) {
            idx_secondNode = itm->second;
        } else {
            r_count++;
            linkMap[secondNode] = r_count;
            idx_secondNode = r_count;
            adj_vector = new vector<int>;
            R_NODES.push_back(adj_vector);
            R_LABELS.push_back(secondNode);
        }

        R_NODES[idx_secondNode]->push_back(idx_firstNode);
        L_NODES[idx_firstNode]->push_back(idx_secondNode);
        m_count++;
    }

    input.close();

    /// Print statistics
    clock_t begin = clock();
    timeBefore = returnMilliSeconds();

    // Update global variables
    N = l_count + r_count;
    M = m_count;
    LN = l_count; // There are one dummy node at the beginning
    RN = r_count; // There are one dummy node at the beginning

    /* allocate memory */
    MAX_SOL = (double *) malloc((N + M) * sizeof(double));
    L_MASK = (int *) malloc(L_NODES.size() * sizeof(int));
    R_MASK = (int *) malloc(R_NODES.size() * sizeof(int));
    ENO_ARR = (int *) malloc((N + M) * sizeof(int));

    for (i = 1; i < L_NODES.size(); i++) {
        ENO_ARR[i - 1] = i - 1;
        L_MASK[i] = 1;
    }

    for (i = 1; i < R_NODES.size(); i++) {
        ENO_ARR[LN + i - 1] = LN + i - 1;
        R_MASK[i] = 1;
    }

    IntPair pr;
    eno = L_NODES.size() + R_NODES.size() - 2;

    for (i = 1; i < L_NODES.size(); i++) {
        for (j = 0; j < (*(L_NODES[i])).size(); j++) {
            pr.x = i;
            pr.y = (*(L_NODES[i]))[j];
            EDGE_MAP[pr] = eno;
            ENO_ARR[eno] = eno;
            eno++;
        }
    }

    /** Initialize Gurobi Variables */
    omp_set_dynamic(0);
    omp_init_lock(&coord_list_lock);
    omp_init_lock(&queue_lock);
    omp_init_lock(&worker_count_lock);

    /** Create environment */
    GRBloadenv(&gurobi_env, nullptr);

    /** Adding first task to Queue*/
    task_queue.push({1, 1, l_count, r_count});

    /** Execute algorithm */
#pragma omp parallel num_threads(THREAD_LIMIT) shared(worker_count, task_queue) default(none)
    {
        execute_tasks();
    }

    /** Free environment */
    if (gurobi_env != nullptr) {
        GRBfreeenv(gurobi_env);
    }

    timeAfter = returnMilliSeconds();
    clock_t end = clock();
    double execution_time = diff_clock(end, begin);

    long int countLSoln = 0;
    long int countRSoln = 0;

    for (i = 0; i < maxLN; i++) {
        isol = (int) MAX_SOL[i];
        if (isol) {
            countLSoln++;
        }
    }
    printf("----\n");
    for (i = maxLN; i < (maxN); i++) {
        isol = (int) MAX_SOL[i];
        if (isol) {
            countRSoln++;
        }
    }

    /// Print statistics to stdout
    cout << "Output file: " << OUTPUT_FILE << endl;
    printf("%5ld\t%5ld\t%5d\t%5f\n", countLSoln, countRSoln, (int) MAX_OBJ_VAL,
            ((double) (timeAfter - timeBefore) / 1000.0));

    /// Print statistics to output file
    FILE *output = fopen(OUTPUT_FILE.c_str(), "a");
    fprintf(output, "Thread Limit: %d\t Input file: %s\n", THREAD_LIMIT, INPUT_FILE.c_str());
    fprintf(output, "Left\tRight\tObjective\tTime\n");
    fprintf(output, "%5ld\t%5ld\t%9d\t%5f\n", countLSoln, countRSoln, (int) MAX_OBJ_VAL,
            ((double) (timeAfter - timeBefore) / 1000.0));
    fprintf(output, "\n");
    fclose(output);
    return 0;
}

void graphviz(const double *solution) {
    ofstream outFile;
    outFile.open("graph.gv");
    int i, j, k, isol;

    // begin graph
    outFile << "graph G {" << endl;

    // left vertices at the same level
    outFile << "  { rank = same ";
    for (i = 1; i < L_LABELS.size(); i++) {
        outFile << " ; " << L_LABELS[i];
    }
    outFile << "  } ;" << endl;

    // right vertices at the same level
    outFile << "  { rank = same ";
    for (i = 1; i < R_LABELS.size(); i++) {
        outFile << " ; " << R_LABELS[i];
    }
    outFile << "  } ;" << endl;

    // draw nodes
    for (i = 1; i < L_NODES.size(); i++) {
        isol = (int) solution[i - 1];
        if (isol) {
            outFile << "   " << L_LABELS[i] << " [style=filled fillcolor=red] ; " << endl;
        }
    }

    for (i = 1; i < R_NODES.size(); i++) {
        isol = (int) solution[LN + i - 1];
        if (isol) {
            outFile << "   " << R_LABELS[i] << " [style=filled fillcolor=red] ; " << endl;
        }
    }

    // draw edges
    k = LN + RN;
    for (i = 1; i < L_NODES.size(); i++) {
        for (j = 0; j < (*(L_NODES[i])).size(); j++) {
            outFile << "  " << L_LABELS[i] << " -- " << R_LABELS[(*(L_NODES[i]))[j]];
            isol = (int) solution[k];
            if (isol) {
                outFile << " [color=red] ";
            }
            outFile << " ; " << endl;
            k++;
        }
    }

    // end graph
    outFile << "}" << endl;
    outFile.close();
    system("dot -Tpng graph.gv > graph.png");
    system("dot -Tsvg graph.gv > graph.svg");
}

int solve_gurobi(int I, int J, int *rmask, int *lmask, int *enoarr, int &newN, int &newLN, int &newRN, int &newM) {
    // Previously global but now local variables
    int *ind = (int *) malloc((N + M) * sizeof(int));
    auto *val = (double *) malloc((N + M) * sizeof(double));
    auto *obj = (double *) malloc((N + M) * sizeof(double));
    auto *vtype = (char *) malloc((N + M) * sizeof(char));
    auto *sol = (double *) malloc((N + M) * sizeof(double));    // solution array that is filled after gurobi solve

    // Previously local variables
    GRBenv *env = gurobi_env;
    GRBmodel *model = nullptr;
    int error;
    int is_opt_successful;

    int optimization_status;
    double objective_value;
    int n, i, j, degree, k;
    IntPair pr;
    map<IntPair, int>::iterator ei;
    char str[10];

    if (!shall_we_solve(I, J, newLN, newRN)) {
        return (0);
    }

    if (env == nullptr) {
        /* Create environment */
        cout << "Create gurobi environment" << endl;
        error = GRBloadenv(&env, nullptr);
        if (error) goto QUIT;
    }

    GRBsetdblparam(env, "Heuristics", 0.05);
    GRBsetintparam(env, "TuneJobs", 32000);
    GRBsetintparam(env, "LogToConsole", 0);
    GRBsetintparam(env, "OutputFlag", 0);
    GRBsetdblparam(env, "TimeLimit", 10.0);

    /* Create an empty model */
    error = GRBnewmodel(env, &model, "biclique", 0, nullptr, nullptr, nullptr, nullptr, nullptr);
    if (error) goto QUIT;

    /* Add variables */
    for (i = 0; i < newN; i++) {
        obj[i] = 0;
        vtype[i] = GRB_BINARY;
    }
    for (i = newN; i < (newN + newM); i++) {
        obj[i] = 1;
        vtype[i] = GRB_BINARY;
    }
    error = GRBaddvars(model, newN + newM, 0, nullptr, nullptr, nullptr, obj, nullptr, nullptr, vtype, nullptr);
    if (error) goto QUIT;

    /* Change objective sense to maximization */
    error = GRBsetintattr(model, GRB_INT_ATTR_MODELSENSE, GRB_MAXIMIZE);
    if (error) goto QUIT;

    /* Integrate new variables */
    error = GRBupdatemodel(model);
    if (error) goto QUIT;

    k = 0;
    /* constraint 2  */
    for (i = 1; i < L_NODES.size(); i++) {
        for (j = 0; j < (*(L_NODES[i])).size(); j++) {
            pr.x = i;
            pr.y = (*(L_NODES[i]))[j];
            ei = EDGE_MAP.find(pr);
            if (ei != EDGE_MAP.end()) {
                ind[0] = i - 1;
                ind[1] = LN + (*(L_NODES[i]))[j] - 1;
                ind[2] = ei->second;
                val[0] = 1;
                val[1] = 1;
                val[2] = -2;
                sprintf(str, "c%d", k++);
            } else {
                cout << "Error(1): edge cannot be found" << endl;
                exit(0);
            }
        }
    }

    /* constraint 3 */
    for (i = 1; i < L_NODES.size(); i++) {
        if (!lmask[i]) continue;
        degree = 0;
        for (j = 0; j < (*(L_NODES[i])).size(); j++) {
            pr.x = i;
            pr.y = (*(L_NODES[i]))[j];
            if (!rmask[pr.y]) continue;
            ei = EDGE_MAP.find(pr);
            if (ei != EDGE_MAP.end()) {
                ind[degree] = enoarr[ei->second];
                val[degree] = 1;
                degree++;
            } else {
                cout << "Error(2): edge cannot be found" << endl;
                exit(0);
            }
        }
        ind[degree] = enoarr[i - 1];
        val[degree] = -J;
        degree++;
        sprintf(str, "c%d", k++);
        error = GRBaddconstr(model, degree, ind, val, GRB_EQUAL, 0, str);
        if (error) goto QUIT;
    }

    /* constraint 4 */
    for (i = 1; i < R_NODES.size(); i++) {
        if (!rmask[i]) continue;
        degree = 0;
        for (j = 0; j < (*(R_NODES[i])).size(); j++) {
            pr.y = i;
            pr.x = (*(R_NODES[i]))[j];
            if (!lmask[pr.x]) continue;
            ei = EDGE_MAP.find(pr);
            if (ei != EDGE_MAP.end()) {
                ind[degree] = enoarr[ei->second];
                val[degree] = 1;
                degree++;
            } else {
                cout << "Error(3): edge cannot be found" << endl;
                exit(0);
            }
        }
        ind[degree] = enoarr[LN + i - 1];
        val[degree] = -I;
        degree++;
        sprintf(str, "c%d", k++);
        error = GRBaddconstr(model, degree, ind, val, GRB_EQUAL, 0, str);
        if (error) goto QUIT;
    }

    /* constraint 5 */
    for (i = 0; i < newLN; i++) {
        ind[i] = i;
        val[i] = 1;
    }
    sprintf(str, "c%d", k++);
    error = GRBaddconstr(model, newLN, ind, val, GRB_EQUAL, I, str);
    if (error) goto QUIT;

    /* constraint 6 */
    for (i = 0; i < newRN; i++) {
        ind[i] = newLN + i;
        val[i] = 1;
    }
    sprintf(str, "c%d", k++);
    error = GRBaddconstr(model, newRN, ind, val, GRB_EQUAL, J, str);
    if (error) goto QUIT;

    /* Optimize model */
    error = GRBoptimize(model);
    if (error) goto QUIT;

    /* Capture solution information */
    error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimization_status);
    if (error) goto QUIT;

    error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &objective_value);
    if (error) goto QUIT;

    error = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, newN + newM, sol);
    if (error) goto QUIT;

    if (optimization_status == GRB_OPTIMAL) {

#pragma omp critical
        {
            if (MAX_OBJ_VAL < objective_value) {
//                printf("Optimal = %d found for I: %d, J: %d\n", (int) objective_value, I, J);
                MAX_OBJ_VAL = objective_value;
                memcpy(MAX_SOL, sol, (newN + newM) * sizeof(double));
                maxN = newN;
                maxM = newM;
                maxLN = newLN;
                maxRN = newRN;
                resultI = I;
                resultJ = J;
            }
        }

        is_opt_successful = 1;
    } else if (optimization_status == GRB_INF_OR_UNBD) {
        printf("Model is infeasible or unbounded for I: %d, J: %d\n", I, J);
        is_opt_successful = 0;
        update_coord_list(I, J);
    } else {
        printf("Optimization was stopped early for I: %d, J: %d\n", I, J);
        is_opt_successful = 0;
        update_coord_list(I, J);
    }

    QUIT:

    /* Error reporting */
    if (error) {
        is_opt_successful = 0;
        update_coord_list(I, J);
    }

    /* Free model */
    if (model != nullptr) {
        GRBfreemodel(model);
    }

    // Free memory allocated inside the function
    free(ind);
    free(val);
    free(obj);
    free(vtype);
    free(sol);

    return (is_opt_successful);
}

void execute_tasks() {
    while (worker_count > 0 || !task_queue.empty()) {
        if (!task_queue.empty()) {
            omp_set_lock(&queue_lock);

            if (task_queue.empty()) {
                omp_unset_lock(&queue_lock);
                continue;
            }

            /** Increase worker counts */
            omp_set_lock(&worker_count_lock);
            worker_count++;
            omp_unset_lock(&worker_count_lock);

            /** Pop task from queue */
            auto temp = task_queue.front();
            task_queue.pop();

            omp_unset_lock(&queue_lock);

            /** Execute Task */
            quad_search(temp.IL, temp.JL, temp.IR, temp.JR);

            /** Decrease worker counts */
            omp_set_lock(&worker_count_lock);
            worker_count--;
            omp_unset_lock(&worker_count_lock);
        }
    }
}

void quad_search(int IL, int JL, int IR, int JR) {
    int I, J;
    int exists;

    if (IL > IR) return;
    if (JL > JR) return;
    I = (IL + IR) / 2;
    J = (JL + JR) / 2;

    exists = 1;
    if (MAX_OBJ_VAL < I * J) {
        // Previously used in multiple places
        int *rmask = (int *) malloc((N + M) * sizeof(double));
        int *lmask = (int *) malloc((N + M) * sizeof(double));
        int *enoarr = (int *) malloc((N + M) * sizeof(double));
        int newRN, newLN, newM, newN;

        mask_nodes(I, J, rmask, lmask, enoarr, newRN, newLN, newM, newN);
        exists = solve_gurobi(I, J, rmask, lmask, enoarr, newRN, newLN, newM, newN);

        free(rmask);
        free(lmask);
        free(enoarr);
    }

    /** Create new tasks */
    omp_set_lock(&queue_lock);
    if (exists) {
        task_queue.push({I + 1, J + 1, IR, JR});
        task_queue.push({IL, J + 1, I, JR});
        task_queue.push({I + 1, JL, IR, J});

    } else {
        task_queue.push({IL, JL, I - 1, J - 1});
        task_queue.push({IL, J, I - 1, JR});
        task_queue.push({I, JL, IR, J - 1});

    }
    omp_unset_lock(&queue_lock);
}

int shall_we_solve(int I, int J, int newLN, int newRN) {
    omp_set_lock(&coord_list_lock);
    list<Coord>::iterator li;

    if (newLN * newRN < I * J) {
        omp_unset_lock(&coord_list_lock);
        return (0);
    }

    omp_unset_lock(&coord_list_lock);
    return (1);
}

void update_coord_list(int I, int J) {
    omp_set_lock(&coord_list_lock);

    list<Coord>::iterator li;
    struct Coord cr = Coord(I, J);

    li = COORD_LIST.begin();
    while (li != COORD_LIST.end()) {
        if (((*li).x >= I) && ((*li).y >= J)) {
            li = COORD_LIST.erase(li);
        }
        li++;
    }
    COORD_LIST.push_front(cr);

    omp_unset_lock(&coord_list_lock);
}

void mask_nodes(int I, int J, int *rmask, int *lmask, int *enoarr, int &newN, int &newLN, int &newRN, int &newM) {
    int i, j;
    int eno, neno;

    // Copy values from input variables
    memcpy(rmask, R_MASK, (N + M) * sizeof(int));
    memcpy(lmask, L_MASK, (N + M) * sizeof(int));
    memcpy(enoarr, ENO_ARR, (N + M) * sizeof(int));
    newN = N;
    newLN = LN;
    newRN = RN;
    newM = M;

    for (i = 1; i < L_NODES.size(); i++) {
        lmask[i] = 0;
        if ((*(L_NODES[i])).size() >= J) {
            lmask[i] = 1;
        }
    }

    for (i = 1; i < R_NODES.size(); i++) {
        rmask[i] = 0;
        if ((*(R_NODES[i])).size() >= I) {
            rmask[i] = 1;
        }
    }

    eno = 0;
    neno = 0;
    for (i = 1; i < L_NODES.size(); i++) {

        if (lmask[i]) {
            enoarr[eno] = neno;
            neno++;
        }
        eno++;
    }
    newLN = neno;

    for (i = 1; i < R_NODES.size(); i++) {
        if (rmask[i]) {
            enoarr[eno] = neno;
            neno++;
        }
        eno++;
    }
    newN = neno;
    newRN = newN - newLN;

    for (i = 1; i < L_NODES.size(); i++) {
        for (j = 0; j < (*(L_NODES[i])).size(); j++) {
            if (lmask[i] && rmask[(*(L_NODES[i]))[j]]) {
                enoarr[eno] = neno;
                neno++;
            }
            eno++;
        }
    }
    newM = neno - newN;
}
