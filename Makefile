INC      =  $(GUROBI_HOME)/include/
CC       =  gcc
CPP      =  clang++
CARGS    = -Xpreprocessor -fopenmp -std=c++14
CLIB     = -L$(GUROBI_HOME)/lib/ -lgurobi91
CPPLIB   = -L$(GUROBI_HOME)/lib/ -lgurobi_c++ -lgurobi91

all:	biip


biip:	biip.cpp
		$(CPP) $(CARGS) -o biip biip.cpp -I$(INC) $(CPPLIB) -lomp -lpthread -lm
