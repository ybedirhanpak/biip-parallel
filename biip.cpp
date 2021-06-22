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

extern "C" {
#include "gurobi_c.h"
}

using namespace std;

// Definitions
typedef struct IntPair {
    int x;
    int y;
} IntPair;

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

int solve_gurobi(int I, int J);

int quad_search(int IL, int JL, int IR, int JR);

int shall_we_solve(int I, int J);

void update_coord_list(int I, int J);

void mask_nodes(int I, int J);

/// Input variables

// Number of nodes, edges
int N, M;
// Number of left nodes, right nodes
int LN, RN;

map<IntPair, int, IntPairCompare> EDGE_MAP;
vector<vector<int> *> L_NODES;
vector<vector<int> *> R_NODES;
vector<string> L_LABELS;
vector<string> R_LABELS;

/// Computational Variables

// Result variables
double maxobjval = 0;
double *maxsol;         // keeps the solution array when the obj val is found maximum

// Used in only gurobi solve
GRBenv *gurobi_env = nullptr;
double *sol;            // solution array that is filled after gurobi solve
int *ind;
double *val;
double *obj;
char *vtype;

// Used in multiple functions, needs to be localized
list<Coord> coord_list; // coordinates list
int *rmask;
int *lmask;
int *enoarr;
int newRN, newLN, newM, newN, maxLN, maxRN, maxN, maxM;

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
    int i, j, eno, isol;
    int rc;

    long int timeBefore;
    long int timeAfter;

    ////////////// Read the graph
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

    while (cin >> firstNode) {
        cin >> secondNode;
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

    /// Print statistics
    clock_t begin = clock();
    timeBefore = returnMilliSeconds();

    // Update global variables
    N = l_count + r_count;
    M = m_count;
    LN = l_count; // There are one dummy node at the beginning
    RN = r_count; // There are one dummy node at the beginning

    /* allocate memory */
    sol = (double *) malloc((N + M) * sizeof(double));
    maxsol = (double *) malloc((N + M) * sizeof(double));
    ind = (int *) malloc((N + M) * sizeof(int));
    val = (double *) malloc((N + M) * sizeof(double));
    obj = (double *) malloc((N + M) * sizeof(double));
    vtype = (char *) malloc((N + M) * sizeof(char));
    lmask = (int *) malloc(L_NODES.size() * sizeof(int));
    rmask = (int *) malloc(R_NODES.size() * sizeof(int));
    enoarr = (int *) malloc((N + M) * sizeof(int));

    for (i = 1; i < L_NODES.size(); i++) {
        enoarr[i - 1] = i - 1;
        lmask[i] = 1;
    }

    for (i = 1; i < R_NODES.size(); i++) {
        enoarr[LN + i - 1] = LN + i - 1;
        rmask[i] = 1;
    }

    IntPair pr;
    eno = L_NODES.size() + R_NODES.size() - 2;

    for (i = 1; i < L_NODES.size(); i++) {
        for (j = 0; j < (*(L_NODES[i])).size(); j++) {
            pr.x = i;
            pr.y = (*(L_NODES[i]))[j];
            EDGE_MAP[pr] = eno;
            enoarr[eno] = eno;
            eno++;
        }
    }

    /* Create environment */
    GRBloadenv(&gurobi_env, nullptr);

    /* Execute algorithm */
    rc = quad_search(1, 1, l_count, r_count);

    /* Free environment */
    if (gurobi_env != nullptr) {
        GRBfreeenv(gurobi_env);
    }

    timeAfter = returnMilliSeconds();
    clock_t end = clock();
    double execution_time = diff_clock(end, begin);

    long int countLSoln = 0;
    long int countRSoln = 0;

    for (i = 0; i < maxLN; i++) {
        isol = (int) maxsol[i];
        if (isol) {
            countRSoln++;
        }
    }
    printf("----\n") ;
    for (i = maxLN; i < (maxN); i++) {
        isol = (int) maxsol[i];
        if (isol) {
            countLSoln++;
        }
    }

    printf("%d\t%ld\t%ld\t%d\t%lf\t%f", 1, countRSoln, countLSoln, rc, execution_time,
            ((double) (timeAfter - timeBefore) / 1000.0));
    FILE *fptr2 = fopen("status.txt", "w");
    fprintf(fptr2, "%d\t%ld\t%ld\t%d\t%lf\t%f", 1, countRSoln, countLSoln, rc, execution_time,
            ((double) (timeAfter - timeBefore) / 1000.0));
    fclose(fptr2);
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

int solve_gurobi(int I, int J) {
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

    if (!shall_we_solve(I, J)) {
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
        if (maxobjval < objective_value) {
            maxobjval = objective_value;
            memcpy(maxsol, sol, (newN + newM) * sizeof(double));
            maxN = newN;
            maxM = newM;
            maxLN = newLN;
            maxRN = newRN;
        }
        is_opt_successful = 1;
    } else if (optimization_status == GRB_INF_OR_UNBD) {
        printf("Model is infeasible or unbounded\n");
        is_opt_successful = 0;
        update_coord_list(I, J);
    } else {
        printf("Optimization was stopped early\n");
        is_opt_successful = 0;
        update_coord_list(I, J);
    }

    QUIT:

    /* Error reporting */
    if (error) {
//        printf("ERROR: %s\n", GRBgeterrormsg(env));
        is_opt_successful = 0;
        update_coord_list(I, J);
    }

    /* Free model */
    if (model != nullptr) {
        GRBfreemodel(model);
    }

//    /* Free environment */
//    if (env != NULL) {
//        GRBfreeenv(env);
//    }

    return (is_opt_successful);
}

int quad_search(int IL, int JL, int IR, int JR) {
    int maxEdges;
    int m1, m2, m3;
    int I, J;
    int exists;

    if (IL > IR) return (0);
    if (JL > JR) return (0);
    I = (IL + IR) / 2;
    J = (JL + JR) / 2;

    exists = 1;
    if (maxobjval < I * J) {
        mask_nodes(I, J);
        exists = solve_gurobi(I, J);
    }

    if (exists) {
        maxEdges = I * J;
        m1 = quad_search(I + 1, J + 1, IR, JR);
        m2 = quad_search(IL, J + 1, I, JR);
        m3 = quad_search(I + 1, JL, IR, J);
    } else {
        maxEdges = 0;
        m1 = quad_search(IL, JL, I - 1, J - 1);
        m2 = quad_search(IL, J, I - 1, JR);
        m3 = quad_search(I, JL, IR, J - 1);
    }

    if (maxEdges < m1) {
        maxEdges = m1;
    }
    if (maxEdges < m2) {
        maxEdges = m2;
    }
    if (maxEdges < m3) {
        maxEdges = m3;
    }
    return (maxEdges);
}

int shall_we_solve(int I, int J) {
    list<Coord>::iterator li;

    if (newLN * newRN < I * J) return (0);
    for (li = coord_list.begin(); li != coord_list.end(); li++) {
        if (((*li).x <= I) && ((*li).y <= J)) return (0);
    }
    return (1);
}

void update_coord_list(int I, int J) {
    list<Coord>::iterator li;
    struct Coord cr = Coord(I, J);

    li = coord_list.begin();
    while (li != coord_list.end()) {
        if (((*li).x >= I) && ((*li).y >= J)) {
            li = coord_list.erase(li);
        }
        li++;
    }
    coord_list.push_front(cr);
}

void mask_nodes(int I, int J) {
    int i, j;
    int eno, neno;
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
