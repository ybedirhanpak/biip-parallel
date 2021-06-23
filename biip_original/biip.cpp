#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <iostream>
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <algorithm>
#include <sys/time.h>

extern "C" {
#include "gurobi_c.h"
}

#define DEBUG_ENABLE 0

using namespace std ;

// Definitions
typedef struct IntPair {
    int  x ;
    int  y ;
} IntPair ;

struct IntPairCompare {
    bool operator()( const IntPair & first, const IntPair & second) const {
        return first.x < second.x || (first.x == second.x && first.y < second.y) ;
    }
} ;

// definitions
struct Coord {
    int x,y ;
} ;

// Global Variables
map< IntPair, int , IntPairCompare > edgemap ;
map< string, int > linkmap ;
vector< vector<int> * > lnodes ;
vector< vector<int> * > rnodes ;
vector<string> llabels ;
vector<string> rlabels ;
int       maxobjval = 0 ;
list<Coord> clist ; // coordinates list
double    *sol;
double    *maxsol;
int       *ind;
double    *val;
double    *obj;
char      *vtype;
int       *rmask ;
int       *lmask ;
int       *enoarr ;
GRBenv *gurobi_env = nullptr;

// Function Prototypes
void graphviz(double *) ;
int  solvegurobi2(int ln,int rn,int m,int L,int K) ;
int  quadsearch(int, int, int, int, int, int, int) ;
int  shallwesolve(int, int) ;
void  updateclist(int,int) ;
void  masknodes(int,int) ;
int  newrn, newln, newm, newn, maxln, maxrn, maxn, maxm ;
int n, m, ln,rn ;


// For calculation of seconds

double diffclock(clock_t clock1,clock_t clock2)
{
    double diffticks=clock1-clock2;
    double diffms=(diffticks*10)/CLOCKS_PER_SEC;
    return diffms;
}

/*
double diffclock(clock_t clock1,clock_t clock2)
{
	double diffticks=clock1-clock2;
	double diffms=(diffticks)/CLOCKS_PER_SEC;
	return diffms;
}
*/

long int returnMiliSeconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
    return ms;
}

void print_list(double* list, int size) {
    cout << "[";
    for (int i = 0; i < size - 1; i++) {
        cout << list[i] << " ";
    }
    cout << list[size - 1] << "]" << endl;
}

int main(int argc,char *argv[])
{
    string INPUT_FILE = "../data/all_gene_disease_associations-gurobi.txt";
    string OUTPUT_FILE = "./biip_original_output.txt";

    if (argc > 1) {
        INPUT_FILE = argv[1];
    }

    if (argc > 2) {
        OUTPUT_FILE = argv[2];
    }

    printf("Input file: %s\n", INPUT_FILE.c_str());

    string x,y ;
    int    lcount,rcount,mcount,ix,iy ;
    map< string, int >::iterator itm ;
    vector<int> * pvec ;
    FILE *fp ;
    IntPair  pr ;

    int i,j,k,eno,isol ;

    int rc ;
    double solnTime = 0;

    long int timeBefore = 0;
    long int timeAfter = 0;

    long int countLSoln = 0;
    long int countRSoln = 0;

//    cout << "/**************************************************/" << endl;
//    cout << "	       GUROBI2'S RUN " 			 << endl;
//    cout << "/**************************************************/" << endl;

    ////////////// read the graph
    lcount = 0 ;
    rcount = 0 ;
    mcount = 0 ;
    lnodes.push_back(NULL) ;
    rnodes.push_back(NULL) ;
    llabels.push_back("") ;
    rlabels.push_back("") ;

    fstream input;
    input.open(INPUT_FILE, ios::in);

    while( input >> x  ) {
        input >> y ;
        itm = linkmap.find(x);
        if( itm != linkmap.end() ) {
            ix = itm->second    ;
        }
        else {
            lcount++ ;
            linkmap[x] = lcount ;
            ix = lcount ;
            pvec = new vector<int>  ;
            lnodes.push_back(pvec) ;
            llabels.push_back(x) ;
        }
        itm = linkmap.find(y);
        if( itm != linkmap.end() ) {
            iy = itm->second    ;
        }
        else {
            rcount++ ;
            linkmap[y] = rcount ;
            iy = rcount ;
            pvec = new vector<int>  ;
            rnodes.push_back(pvec) ;
            rlabels.push_back(y) ;
        }

        rnodes[iy]->push_back(ix) ;
        lnodes[ix]->push_back(iy) ;
        mcount++ ;
    }
//    cout << "/**************************************************/" << endl;
//    cout << "	       COMPLETE FILE READING                    " << endl;
//    cout << "/**************************************************/" << endl;

    input.close();

    clock_t begin=clock();
    timeBefore = returnMiliSeconds();

    //////////// print statistics
#ifdef DEBUG_ENABLE
//    cout << lcount << " " << rcount << " " << mcount << endl ;
#endif
    n = lnodes.size() + rnodes.size() - 2  ;
    m = mcount ;
    ln =  lnodes.size() - 1 ;
    rn =  rnodes.size() - 1 ;

    /* allocate memory */
    sol      = (double *)   malloc((n+m)*sizeof(double)) ;
    maxsol   = (double *)   malloc((n+m)*sizeof(double)) ;
    ind      = (int *)      malloc((n+m)*sizeof(int)) ;
    val      = (double *)   malloc((n+m)*sizeof(double)) ;
    obj      = (double *)   malloc((n+m)*sizeof(double)) ;
    vtype    = (char *)     malloc((n+m)*sizeof(char)) ;
    lmask    = (int *)      malloc(lnodes.size()*sizeof(int)) ;
    rmask    = (int *)      malloc(rnodes.size()*sizeof(int)) ;
    enoarr   = (int *)      malloc((n+m)*sizeof(int)) ;

    for(i=1 ; i < lnodes.size() ; i++) {
        enoarr[i-1] = i-1 ;
        lmask[i] = 1 ;
    }
    for(i=1 ; i < rnodes.size() ; i++) {
        enoarr[ln+i-1] = ln+i-1 ;
        rmask[i] = 1 ;
    }
    eno = lnodes.size() + rnodes.size() - 2  ;
    for(i=1 ; i < lnodes.size() ; i++) {
        for(j=0 ; j < (*(lnodes[i])).size() ; j++) {
            pr.x = i  ;  pr.y = (*(lnodes[i]))[j] ;
            edgemap[pr] = eno ;
            enoarr[eno] = eno ;
            // cout << "edge " <<  pr.x << " " << pr.y << " " << eno << endl ;
            eno++ ;
        }
    }

    //solvegurobi2(lcount,rcount,mcount,3,3) ;


    /* Create environment */
    GRBloadenv(&gurobi_env, nullptr);

    /* Execute algorithm */
    rc = quadsearch(lcount,rcount,mcount,1,1,lcount,rcount) ;

    /* Free environment */
    if (gurobi_env != nullptr) {
        GRBfreeenv(gurobi_env);
    }

//    printf("COMPLETED MAXEDGES: %d\n",rc) ;
//    cout << lcount << " " << rcount << " " << mcount << endl ;
    //masknodes(2,2) ;
    //solvegurobi2(lcount,rcount,mcount,2,2) ;
//    cout << "maxln: " << maxln << ", maxn: " << maxn << endl;

    timeAfter = returnMiliSeconds();
    clock_t end=clock();
    solnTime = double(diffclock(end,begin));


    for(i=0 ; i < maxln ; i++) {
        isol = maxsol[i] ;
        if (isol) {
            //printf("[%2d] %d\n",i,isol);
            countLSoln++;
        }
    }
    printf("----\n") ;
    for(i=maxln ; i < (maxn) ; i++) {
        isol = maxsol[i] ;
        if (isol) {
            //printf("[%2d] %d\n",i,isol);
            countRSoln++;
        }
    }

    /// Print statistics to stdout
    cout << "Output file: " << OUTPUT_FILE << endl;
    printf("%5ld\t%5ld\t%5d\t%5f\n", countLSoln, countRSoln, rc,
           ((double) (timeAfter - timeBefore) / 1000.0));

    /// Print statistics to output file
    FILE *output = fopen(OUTPUT_FILE.c_str(), "a");
    fprintf(output, "Input file: %s\n", INPUT_FILE.c_str());
    fprintf(output, "Left\tRight\tObjective\tTime\n");
    fprintf(output, "%5ld\t%5ld\t%9d\t%5f\n", countLSoln, countRSoln, (int) rc,
            ((double) (timeAfter - timeBefore) / 1000.0));
    fprintf(output, "\n");
    fclose(output);

    return 0;
}

void graphviz(double * sol)
{
    ofstream myfile;
    myfile.open("graph.gv");
    int i,j,k,isol ;
    int  ln, rn ;

    ln = lnodes.size()-1 ;
    rn = rnodes.size()-1 ;

    // begin graph
    myfile << "graph G {" << endl ;

    // left vertices at the same level
    myfile << "  { rank = same " ;
    for(i=1 ; i < llabels.size() ; i++) {
        myfile << " ; " << llabels[i] ;
    }
    myfile << "  } ;" << endl ;


    // right vertices at the same level
    myfile << "  { rank = same "  ;
    for(i=1 ; i < rlabels.size() ; i++) {
        myfile << " ; " << rlabels[i] ;
    }
    myfile << "  } ;" << endl ;

    // draw nodes
    for(i=1 ; i < lnodes.size() ; i++) {
        isol = sol[i-1] ;
        if (isol) {
            myfile << "   " << llabels[i] << " [style=filled fillcolor=red] ; " << endl ;
        }
    }
    for(i=1 ; i < rnodes.size() ; i++) {
        isol = sol[ln+i-1] ;
        if (isol) {
            myfile << "   " << rlabels[i] << " [style=filled fillcolor=red] ; " << endl ;
        }
    }

    // draw edges
    k = ln+rn ;
    for(i=1 ; i < lnodes.size() ; i++) {
        for(j=0 ; j < (*(lnodes[i])).size() ; j++) {
            myfile << "  " << llabels[i] << " -- " << rlabels[(*(lnodes[i]))[j]] ;
            isol = sol[k] ;
            if (isol) {
                myfile << " [color=red] " ;
            }
            myfile << " ; " << endl ;
            k++ ;
        }
    }

    // end graph
    myfile << "}" << endl ;
    myfile.close();
    system("dot -Tsvg graph.gv > graph.svg");
    system("dot -Tpng graph.gv > graph.png");
}

int solvegurobi2(
        int ln,
        int rn,
        int m,
        int I,
        int J)
{
    GRBenv   *env = gurobi_env;
    GRBmodel *model = NULL;
    int       error = 0;
    int       rc ;

    int       optimstatus;
    double    objval;
    int       n,i,j,degree,k ;
    IntPair   pr ;
    map< IntPair, int >::iterator ei ;
    char      str[10] ;
    int       isol ;

    n = ln + rn ;
    rc = 0 ;

    if ( ! shallwesolve(I,J) ) {
        return(0) ;
    }

    if (env == nullptr) {
        /* Create environment */
        cout << "Create gurobi environment" << endl;
        error = GRBloadenv(&env, nullptr);
        if (error) goto QUIT;
    }

    GRBsetdblparam(env, "Heuristics", 0.05 );
    GRBsetintparam(env, "TuneJobs", 32000 );
    GRBsetintparam(env, "LogToConsole", 0 );
    GRBsetintparam(env, "OutputFlag", 0 );
    GRBsetdblparam(env, "TimeLimit", 10.0);

    /* Create an empty model */
    error = GRBnewmodel(env, &model,"biclique", 0, NULL, NULL, NULL, NULL, NULL);
    if (error) goto QUIT;

    //error = GRBsetintparam(env, GRB_INT_PAR_THREADS,8);
    //if (error) goto QUIT;

    /* Add variables */
    for(i=0 ; i < newn ; i++) {
        obj[i] = 0 ;
        vtype[i] = GRB_BINARY;
    }
    for(i=newn ; i < (newn+newm) ; i++) {
        obj[i] = 1 ;
        vtype[i] = GRB_BINARY;
    }
    error = GRBaddvars(model, newn+newm, 0, NULL, NULL, NULL, obj, NULL, NULL, vtype,NULL);
    if (error) goto QUIT;

    /* Change objective sense to maximization */
    error = GRBsetintattr(model, GRB_INT_ATTR_MODELSENSE, GRB_MAXIMIZE);
    if (error) goto QUIT;

    /* Integrate new variables */
    error = GRBupdatemodel(model);
    if (error) goto QUIT;

    k = 0 ;
    /* constraint 2  */
    for(i=1 ; i < lnodes.size() ; i++) {
        for(j=0 ; j < (*(lnodes[i])).size() ; j++) {
            pr.x = i  ;  pr.y = (*(lnodes[i]))[j] ;
            ei = edgemap.find(pr);
            if( ei != edgemap.end() ) {
                ind[0] = i-1 ; ind[1] = ln + (*(lnodes[i]))[j] - 1 ;  ind[2] = ei->second ;
                val[0] = 1   ; val[1] = 1; val[2] = -2 ;
                sprintf(str,"c%d",k++) ;
                error = GRBaddconstr(model,3,ind,val,GRB_GREATER_EQUAL,0,str);
//          if (error) goto QUIT;
            }
            else {
                cout << "Error(1): edge cannot be found" << endl ;
                exit(0) ;
            }
        }
    }
    /*************/


    /* constraint 3 */
    for(i=1 ; i < lnodes.size() ; i++) {
        if (! lmask[i] ) continue ;
        degree = 0 ;
        for(j=0 ; j < (*(lnodes[i])).size() ; j++) {
            pr.x = i  ;  pr.y = (*(lnodes[i]))[j] ;
            if (! rmask[pr.y] ) continue ;
            ei = edgemap.find(pr);
            if( ei != edgemap.end() ) {
                ind[degree] = enoarr[ei->second] ;
                val[degree] = 1   ;
                degree++ ;
            }
            else {
                cout << "Error(2): edge cannot be found" << endl ;
                exit(0) ;
            }
        }
        ind[degree] = enoarr[i - 1] ;
        val[degree] = -J ;
        degree++ ;
        sprintf(str,"c%d",k++) ;
        error = GRBaddconstr(model,degree,ind,val,GRB_EQUAL,0,str);
        if (error) goto QUIT;
    }

#ifndef DEBUG_ENABLE
    printf("passed 1\n") ;
#endif

    /* constraint 4 */
    for(i=1 ; i < rnodes.size() ; i++) {
        if (! rmask[i] ) continue ;
        degree = 0 ;
        for(j=0 ; j < (*(rnodes[i])).size() ; j++) {
            pr.y =  i   ;  pr.x = (*(rnodes[i]))[j] ;
            if (! lmask[pr.x] ) continue ;
            ei = edgemap.find(pr);
            if( ei != edgemap.end() ) {
                ind[degree] = enoarr[ei->second] ;
                val[degree] = 1   ;
                degree++ ;
            }
            else {
                cout << "Error(3): edge cannot be found" << endl ;
                exit(0) ;
            }
        }
        ind[degree] = enoarr[ln + i - 1] ;
        val[degree] = -I ;
        degree++ ;
        sprintf(str,"c%d",k++) ;
        error = GRBaddconstr(model,degree,ind,val,GRB_EQUAL,0,str);
        if (error) goto QUIT;
    }

#ifndef DEBUG_ENABLE
    printf("passed 2\n") ;
#endif

    /* constraint 5 */
    for(i=0 ; i < newln ; i++) {
        ind[i] = i ;
        val[i] = 1 ;
    }
    sprintf(str,"c%d",k++) ;
    error = GRBaddconstr(model,newln,ind,val,GRB_EQUAL,I,str);
    if (error) goto QUIT;

#ifndef DEBUG_ENABLE
    printf("passed 3\n") ;
#endif

    /* constraint 6 */
    for(i=0 ; i < newrn ; i++) {
        ind[i] = newln+i ;
        val[i] = 1 ;
    }
    sprintf(str,"c%d",k++) ;
    error = GRBaddconstr(model,newrn,ind,val,GRB_EQUAL,J,str);
    if (error) goto QUIT;

#ifndef DEBUG_ENABLE
    printf("passed 4\n") ;
#endif

    //GRBsetintparam(env, "ConcurrentMIP", 8);

    /* Optimize model */
    error = GRBoptimize(model);
    if (error) goto QUIT;

#ifndef DEBUG_ENABLE
    printf("passed 5\n") ;
#endif

    ///* Write model to 'mip1.lp' */
    //error = GRBwrite(model, "biclique.lp");
    //if (error) goto QUIT;

    /* Capture solution information */
    error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
    if (error) goto QUIT;

#ifndef DEBUG_ENABLE
    printf("passed 6\n") ;
#endif

    error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &objval);
    if (error) goto QUIT;

#ifndef DEBUG_ENABLE
    printf("passed 7\n") ;
#endif

    error = GRBgetdblattrarray(model, GRB_DBL_ATTR_X,0,newn+newm,sol);
    if (error) goto QUIT;

#ifndef DEBUG_ENABLE
    printf("passed 8\n") ;
#endif

//    printf("\nOptimization complete\n");
    if (optimstatus == GRB_OPTIMAL) {
//        printf("Optimal objective: %.0f\n", objval);
        if (maxobjval < objval) {
            printf("Optimal = %d found for I: %d, J: %d\n", (int) objval, I, J);
            maxobjval = objval ;
            memcpy (maxsol,sol,(newn+newm)*sizeof(double));
            maxn = newn ;
            maxm = newm ;
            maxln = newln ;
            maxrn = newrn ;
        }
        rc = 1 ;
    } else if (optimstatus == GRB_INF_OR_UNBD) {
        printf("Model is infeasible or unbounded for I: %d, J: %d\n", I, J);
        rc = 0 ;
        updateclist(I,J) ;
    } else {
        printf("Optimization was stopped early for I: %d, J: %d\n", I, J);
        rc = 0 ;
        updateclist(I,J) ;
    }
#ifndef DEBUG_ENABLE
    printf("passed 9\n") ;
#endif
    QUIT:

    /* Error reporting */
    if (error) {
//        printf("ERROR: %s\n", GRBgeterrormsg(env));
        rc = 0 ;
        updateclist(I,J) ;
    }

    /* Free model */
    if (model != NULL) {
        GRBfreemodel(model);
    }

//    /* Free environment */
//    if (env != NULL) {
//        GRBfreeenv(env);
//    }

#ifndef DEBUG_ENABLE
    printf("SOLVE: %d %d %d\n",I,J,rc) ;
#endif

    return(rc);
}

int quadsearch(
        int ln,
        int rn,
        int m,
        int IL,
        int JL,
        int IR,
        int JR)
{
    int maxedges ;
    int m1,m2,m3  ;
    int I,J ;
    int exists ;

//    printf("quadsearch(%d,%d,%d,%d)\n",IL,JL,IR,JR) ;
    if ( IL > IR) return(0) ;
    if ( JL > JR) return(0) ;
    I = (IL+IR) / 2 ;
    J = (JL+JR) / 2 ;

    exists = 1 ;
    if ( maxobjval < I*J ) {
        masknodes(I,J) ;
        exists = solvegurobi2(ln,rn,m,I,J) ;
    }
    if ( exists ) {
//        printf("quad_search\t \t(%d, %d, %d, %d), (%d, %d) = %d\n", IL, JL, IR, JR, I, J, I * J);
        maxedges = I*J ;

#ifndef DEBUG_ENABLE
        printf("MAXEDGES: %d\n",maxedges) ;
#endif

        m1 = quadsearch(ln,rn,m,I+1,J+1,IR,JR) ;
        m2 = quadsearch(ln,rn,m,IL,J+1,I,JR) ;
        m3 = quadsearch(ln,rn,m,I+1,JL,IR,J) ;
    }
    else {
//        printf("quad_search\t \t(%d, %d, %d, %d), (%d, %d) = %d\n", IL, JL, IR, JR, I, J, 0);
        maxedges = 0 ;
        m1 = quadsearch(ln,rn,m,IL,JL,I-1,J-1) ;
        m2 = quadsearch(ln,rn,m,IL,J,I-1,JR) ;
        m3 = quadsearch(ln,rn,m,I,JL,IR,J-1) ;
    }

    if (maxedges < m1 ) {
        maxedges = m1 ;
    }
    if (maxedges < m2 ) {
        maxedges = m2 ;
    }
    if (maxedges < m3 ) {
        maxedges = m3 ;
    }
    return(maxedges) ;
}

int shallwesolve(
        int I,
        int J)
{
    list<Coord>::iterator li ;

    if (newln*newrn < I*J) {
//        printf("shall_we_solve\t(%d, %d, %d, %d) = false \n", I, J, newln, newrn);
        return (0);
    }
    for(li = clist.begin(); li != clist.end(); li++) {
        if ( ((*li).x <= I) &&  ((*li).y <= J) ) {
//            printf("shall_we_solve\t(%d, %d, %d, %d) = false \n", I, J, newln, newrn);
            return(0) ;
        }
    }

//    printf("shall_we_solve\t(%d, %d, %d, %d) = true \n", I, J, newln, newrn);
    return(1) ;
}

void updateclist(
        int I,
        int J)
{
    list<Coord>::iterator li ;
    Coord cr ;

    cr.x = I ;
    cr.y = J ;
    li = clist.begin() ;
    while(li  != clist.end() ) {
        if ( ((*li).x >= I) &&  ((*li).y >= J) ) {
            li = clist.erase(li);
        }
        li++ ;
    }
    clist.push_front(cr);
}

void masknodes(
        int I,
        int J)
{
    int i,j ;
    int eno,neno ;

    newn = n ;
    newln = ln ;
    newrn = rn ;
    newm = m ;

    for(i=1 ; i < lnodes.size() ; i++) {
        lmask[i] = 0 ;
        if ( (*(lnodes[i])).size() >= J ) {
            lmask[i] = 1 ;
        }
    }

    for(i=1 ; i < rnodes.size() ; i++) {
        rmask[i] = 0 ;
        if ( (*(rnodes[i])).size() >= I) {
            rmask[i] = 1 ;
        }
    }

    //cout << "I:" << I << "-J:" << J << endl;
/*
    for(i=1 ; i < lnodes.size() ; i++) {

	 int countUnMasked = 0;
	 for(int k=1 ; k < (*(lnodes[i])).size() ; k++){
	    if ( !lmask[(*(lnodes[i]))[k]] )
	    {
		countUnMasked++;
	    }
	 }

	 if ( countUnMasked <= J )  {
           //cout << i << " masked cause " << countUnMasked << " <" << J << endl;
           lmask[i] = 0;
         }
	 else
	 {
	   //cout << i << " unmasked cause " << countUnMasked << " <" << J << endl;
	 }
    }
*/
/*
    for(i=1 ; i < rnodes.size() ; i++) {

         int countUnMasked = 0;
         for(int k=1 ; k < (*(rnodes[i])).size() ; k++){
            if ( !lmask[(*(rnodes[i]))[k]] )
            {
                countUnMasked++;
            }
         }

         if ( countUnMasked <= I )  {
           //cout << i << " masked cause " << countUnMasked << " <" << J << endl;
           rmask[i] = 0;
         }
         else
         {
           //cout << i << " unmasked cause " << countUnMasked << " <" << J << endl;
         }
    }
 */
    eno = 0 ;
    neno=0 ;
    for(i=1 ; i < lnodes.size() ; i++) {

        if ( lmask[i] ) {
            enoarr[eno] = neno ;
            neno++ ;
        }
        eno++ ;
    }
    newln = neno ;

    for(i=1 ; i < rnodes.size() ; i++) {
        if ( rmask[i] ) {
            enoarr[eno] = neno ;
            neno++ ;
        }
        eno++ ;
    }
    newn  = neno ;
    newrn = newn - newln ;


    for(i=1 ; i < lnodes.size() ; i++) {
        for(j=0 ; j < (*(lnodes[i])).size() ; j++) {
            if ( lmask[i] &&  rmask[(*(lnodes[i]))[j]] )  {
                enoarr[eno] = neno ;
                neno++ ;
            }
            eno++ ;
        }
    }
    newm = neno - newn ;
}
