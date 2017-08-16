**************************************************************
Parallel Implementation of Decision Tree Using OpenMP and CUDA
**************************************************************
Team Members:-
Bhushan Pagariya (bpagariy@uci.edu) - 76245890, 
Jatin Mehta (jatinm@uci.edu) - 20807196



*************************************************
Steps to run CUDA Implementation of Decision Tree
*************************************************
1. Load required module :
	module load  cuda/5.0
	module load  gcc/4.4.3
2. Complile project :
	nvcc -arch compute_20 project.cu timer.c DecisionTreeCuda.cpp -o project
3. Make changes in 'cuda_job_sub.sh' and provide dataset size and gpu queue information
4. Submit job 'cuda_job_sub.sh' to GPU queue


***************************************************
Steps to run OpenMP Implementation of Decision Tree
***************************************************
1. Complile project :
	g++ -fopenmp -std=c++0x openmp_DecisionTree.cpp -o openmp_DecisionTree.o
2. Make changes in 'openmp_job_sub.sh' and provide dataset size and queue information
3. Submit job 'openmp_job_sub.sh' to queue

*******************************************************
Steps to run Sequential Implementation of Decision Tree
*******************************************************
1. Complile project :
	g++ sequential_decisionTree.cpp -std=c++0x -o sequential_decisionTree.o
2. Run object file with dataset size:
	./sequential_decisionTree.o 100000 784