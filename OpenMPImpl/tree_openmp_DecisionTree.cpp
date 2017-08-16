//
// Created by Bhushan Pagariya on 6/1/17.
//

#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <algorithm>
#include<math.h>
#include <omp.h>
#include<cstring>
#include<cstdlib>

#include<time.h>


#include <chrono>
using namespace std::chrono;


using namespace std;

struct Node {
    int attributeId;
    float threshold;
    int classInd;
    float proportion;
    Node * leftChild;
    Node * rightChild;
    Node(int attributeId, float threshold){
        this->attributeId = attributeId;
        this->threshold = threshold;
        this->classInd = -1;
        this->proportion = -1;
        this->leftChild = NULL;
        this->rightChild = NULL;
    }
    Node(int attributeId, float threshold, int classInd, float proportion){
        this->attributeId = attributeId;
        this->threshold = threshold;
        this->classInd = classInd;
        this->proportion = proportion;
        this->leftChild = NULL;
        this->rightChild = NULL;
    }
};

//void gpuHandler(float * samplesData, int * targetVals, int numSamples, int numAttrs, int numClasses, int * remAttributes, int & maxAttr, float & maxThreshold);

//void readCsv(string filename, vector<vector<float> >& ans);
/*
double getEntropy(vector<int>& quality_score, int numOfClasses){
    int data_points = quality_score.size();
    vector<float> count(numOfClasses, 0);
    int i;
    for(i = 0; i < data_points; i++)
        count[quality_score[i]]++;
    double entropy = 0;
    for(int i=0; i<numOfClasses; i++){
        float proportion = count[i]/data_points;
        if(proportion!=0)
            entropy += (-1)*proportion*log2(proportion);
    }

    return entropy;
}
*/

float getThresholdSplit(float * data_mat, int numSamples, int numAttrs, int attribute){
//    float attribute_column[numSamples];
//    for(int i=0; i<numSamples; i++)
//        attribute_column[i]=data_mat[i*numAttrs+attribute];
//    sort(attribute_column, attribute_column+numSamples);
//    return attribute_column[numSamples/2];
    float max = 0;
    float min = 100000;
    for(int i = 0; i<numSamples; i++) {
        if(data_mat[i*numAttrs+attribute]>max)
            max = data_mat[i*numAttrs+attribute];
        if(data_mat[i*numAttrs+attribute]<min)
            min = data_mat[i*numAttrs+attribute];
    }
    return (max+min)/2;


}

float computeEntropy(int * classHisto, int numClasses, int numSamples) {
    float entropy = 0;
    float proportion = 0;
    for(int i=0; i<numClasses; i++){
        //printf("class count-  %d ", classHisto[i]);
        if(classHisto[i] == 0)
            continue;
        proportion = ((float)classHisto[i])/numSamples;
        entropy += (-1)*proportion*log2(proportion);
    }
    return entropy;
}

void computeInfoGain(float * data_mat, int * targetValues, int start, int numSamples, int numAttrs, int attribute, float * informationGain, float * threshold, int numOfClasses){
    threshold[0] = getThresholdSplit(data_mat+start*numAttrs, numSamples, numAttrs, attribute);
    //printf("Threshold for attr %d is %f\n",attribute, threshold);
    int parentClassCount[numOfClasses];
    for(int i = 0; i<numOfClasses; i++)
        parentClassCount[i] = 0;
    int leftChildClassCount[numOfClasses];
    for(int i = 0; i<numOfClasses; i++)
        leftChildClassCount[i] = 0;
    int rightChildClassCount[numOfClasses];
    for(int i = 0; i<numOfClasses; i++)
        rightChildClassCount[i] = 0;
    int leftSampleCount = 0;
    int rightSampleCount = 0;
    for(int i = start ; i<start+numSamples; i++) {
        if(data_mat[i*numAttrs+attribute] <= threshold[0]) {
            leftChildClassCount[targetValues[i]]++;
            leftSampleCount++;
        } else {
            rightChildClassCount[targetValues[i]]++;
            rightSampleCount++;
        }
        parentClassCount[targetValues[i]]++;
    }
    if(leftSampleCount==0 || rightSampleCount==0){
        informationGain = 0;
        return;
    }


    float entropy_left = computeEntropy(leftChildClassCount, numOfClasses, leftSampleCount);
    float entropy_right = computeEntropy(rightChildClassCount, numOfClasses, rightSampleCount);
    float entropy_parent = computeEntropy(parentClassCount, numOfClasses, numSamples);
    //printf("Entropy left - %f, right - %f, parent - %f, numSamples - %d\n", entropy_left, entropy_right, entropy_parent, numSamples);
    float avgEntropy = (leftSampleCount/(numSamples))*entropy_left + (rightSampleCount/(numSamples))*entropy_right;
    informationGain[0] = entropy_parent - avgEntropy;
}

/*
void computeInfoGain_all(vector<vector<float> > &data_mat, vector<int> targetValues, vector<int> attributes, vector<float> &informationGains, vector<float> &thresholds, int numOfClasses){
    float informationGain, threshold;
    for(int i = 0; i < attributes.size() ; i++) {
        computeInfoGain(dataMat, targetVal, i, informationGain, threshold, numOfClasses);       
        informationGains[i] = informationGains;
        thresholds[i] = threshold;
    }
}
*/

Node * buildDecisionTree(float * dataMat, float * dataMat_bkp, int * targetVal, int * targetVal_bkp, int start, int numSamples, int numAttrs, int leafEntriesThreshold, int * remainingAttrs, int numOfClasses, int numThreads) {
    if(numSamples < leafEntriesThreshold) {
        // No information gain after splitting on any attribute
        int countPerClass[numOfClasses];
        // Initialize to 0
        for(int i = 0; i<numOfClasses; i++)
            countPerClass[i] = 0;

        for(int i = start; i<start+numSamples; i++)
            countPerClass[targetVal[i]]++;
        int max = 0;
        int index = -1;
        for(int i = 0; i<numOfClasses; i++)
            if(countPerClass[i]>max) {
                max = countPerClass[i];
                index = i;
            }

        return new Node(-1, -1, index, (float)max/numSamples);
    }

    float maxInfoGain=0, maxThreshold=0;
    int maxAttr = -1;
    // Compute Information Gain over all attributes
//    float informationGain, threshold;
    int i;
    float infoGainArr[numAttrs];
    for(int j = 0; j<numAttrs; j++)
        infoGainArr[j] = -1;
    float thresholdArr[numAttrs];
    for(int j = 0; j<numAttrs; j++)
        thresholdArr[j] = -1;

//gpuHandler(dataMat+start*numAttrs, targetVal+start, numSamples, numAttrs, numOfClasses, remainingAttrs, maxAttr, maxThreshold);

    //#pragma omp parallel for shared (dataMat, targetVal, start, numAttrs, numSamples, numOfClasses, infoGainArr, thresholdArr) private(i)
    //#pragma omp parallel for

    #pragma omp parallel for private (remainingAttrs, infoGainArr, thresholdArr, dataMat, targetVal, i, start, numSamples, numAttrs, numOfClasses)
    for(i = 0; i< numAttrs; i++){
            //float informationGain, threshold;
            if(remainingAttrs[i]!=0)
                continue;
            computeInfoGain(dataMat, targetVal, start, numSamples, numAttrs, i, infoGainArr+i, thresholdArr+i, numOfClasses);
            //infoGainArr[i] = informationGain;
            //thresholdArr[i] = threshold;
    }
    #pragma omp barrier


    for(int i = 0; i<numAttrs; i++) {
        if(infoGainArr[i] > maxInfoGain) {
            maxInfoGain = infoGainArr[i];
            maxThreshold = thresholdArr[i];
            maxAttr = i;
        }
    }   

    if(maxAttr == -1) {

        int countPerClass[numOfClasses];
        for(int i = 0; i<numOfClasses; i++)
            countPerClass[i] = 0;
        for(int i = 0; i<numSamples; i++)
            countPerClass[targetVal[i]]++;
        int max = 0;
        int index = -1;
        for(int i = 0; i<numOfClasses; i++)
            if(countPerClass[i]>max) {
                max = countPerClass[i];
                index = i;
            }

        return new Node(-1, -1, index, (float)max/numSamples);
    }

    remainingAttrs[maxAttr] = 1;
    int remForLeft[numAttrs], remForRight[numAttrs];
    memcpy(remForLeft, remainingAttrs, numAttrs*sizeof(int));
    memcpy(remForRight, remainingAttrs, numAttrs*sizeof(int));
    // Create Node
    Node * node = new Node(maxAttr, maxThreshold);

    //Split on attribute having max information gain
    
    int leftValueCount = 0;
    int rightValueCount = 0;
    for(int i=start; i<start+numSamples; i++) {
        if (dataMat[i*numAttrs+maxAttr] <= maxThreshold) {
            memcpy(dataMat+leftValueCount*numAttrs, dataMat+i*numAttrs, numAttrs*sizeof(float));
            targetVal[leftValueCount] = targetVal[i];
            leftValueCount++;
        } else {
            memcpy(dataMat_bkp+rightValueCount*numAttrs, dataMat+i*numAttrs, numAttrs*sizeof(float));
            targetVal_bkp[rightValueCount] = targetVal[i];
            rightValueCount++;
        }
    }
    //printf("Left Partition - %d, Right artition - %d\n", leftValueCount, rightValueCount);
    // merging both parts

    memcpy(dataMat+leftValueCount*numAttrs, dataMat_bkp, rightValueCount*numAttrs*sizeof(float));
    memcpy(targetVal+leftValueCount,targetVal_bkp,rightValueCount*sizeof(float));

     if(numThreads <= 1) {
        node->leftChild = buildDecisionTree(dataMat, dataMat_bkp, targetVal, targetVal_bkp, 0, leftValueCount, numAttrs, leafEntriesThreshold, remForLeft, numOfClasses, numThreads);
        node->rightChild = buildDecisionTree(dataMat+leftValueCount*numAttrs, dataMat_bkp+leftValueCount*numAttrs, targetVal+leftValueCount, targetVal_bkp+leftValueCount, 0, rightValueCount, numAttrs, leafEntriesThreshold, remForRight, numOfClasses, numThreads);
     } else {
        #pragma omp task shared(dataMat, dataMat_bkp, targetVal, targetVal_bkp)
        {
            node->leftChild = buildDecisionTree(dataMat, dataMat_bkp, targetVal, targetVal_bkp, 0, leftValueCount, numAttrs, leafEntriesThreshold, remForLeft, numOfClasses, numThreads/2);
        }
        node->rightChild = buildDecisionTree(dataMat+leftValueCount*numAttrs, dataMat_bkp+leftValueCount*numAttrs, targetVal+leftValueCount, targetVal_bkp+leftValueCount, 0, rightValueCount, numAttrs, leafEntriesThreshold, remForRight, numOfClasses, numThreads/2);   
        #pragma omp taskwait
    }


    return node;
}


void printDecisionTree(Node * root, string str, int & count) {
    if(!root)
        return;
    printDecisionTree(root->leftChild, str+"\t\t\t", count);
 //   cout<<str<<"("<<root->attributeId<<","<<root->threshold<<","<<root->classInd<<","<<root->proportion<<")"<<endl;
    count++;
    printDecisionTree(root->rightChild, str+"\t\t\t", count);
}


void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  cout<<"Elapsed time: "<<elapsed<<" s\n";
}




int main() {
//    vector<vector<float> > data_mat;
//    readCsv("/data/users/bpagariy/mnist_train.csv", data_mat);
  
//    cout<<"Num of Samples:- "<<data_mat.size()<<endl;
//    cout<<"Num of Attr:- "<<data_mat[0].size()<<endl;
//    int numSamples = data_mat.size();
//    int attrCount = data_mat[0].size()-1;


    /* timer */
    clock_t start, stop;
    

    int numSamples = 5000;
    int attrCount = 74;
 
    float * samples = (float *) malloc(numSamples*attrCount*sizeof(float));
    float * dataMat_bkp = (float *) malloc(numSamples*attrCount*sizeof(float));
    int offset = 0;
    for(int i = 0; i < numSamples; i++)
        for(int j = 0; j<attrCount; j++) {
           // samples[offset] = data_mat[i][j];
            samples[offset] = (float)(rand()%1000)/10;
            offset++;
        }
    //int numSamples = data_mat.size();
    //int attrCount = data_mat[0].size()-1;
    //cout<<"Samples:- "<<endl;
    //for(int i = 0; i<numSamples*attrCount; i++)
    //    cout<<samples[i]<<" ";
    //cout<<"\n----------------\n";
    
    //printf("Target Values:- \n");
    int * targetVals = (int *) malloc (numSamples*sizeof(int));
    int * targetVal_bkp = (int *) malloc (numSamples*sizeof(int));
    for(int i = 0; i<numSamples ; i++){
        //targetVals[i] = (int) data_mat[i][0];
        targetVals[i] = rand()%10;
        //printf("%d", targetVals[i]);
    }
    //printf("\n----------------\n");

    int * allAvailableAttrs = (int *) malloc (attrCount*sizeof(int));
    for(int i = 0; i<attrCount; i++)
        allAvailableAttrs[i] = 0;
    
    
    /* initialize timer */
    start = clock();


high_resolution_clock::time_point t1 = high_resolution_clock::now();

Node * rootNode = NULL;
#pragma omp parallel
    {
        #pragma omp single nowait
        {
        rootNode=buildDecisionTree(samples, dataMat_bkp, targetVals, targetVal_bkp, 0, numSamples, attrCount, 10, allAvailableAttrs, 10, omp_get_thread_num());   
        }
    }

high_resolution_clock::time_point t2 = high_resolution_clock::now();

auto duration = duration_cast<microseconds>( t2 - t1 ).count();

    cout <<"Chrono time:- "<<duration<<endl;


    cout<<"h"<<endl;
    stop = clock();
    print_elapsed(start, stop);

    int temp = 0;
    printDecisionTree(rootNode, "", temp);
    cout<<temp<<endl;
    cout<<"Done"<<endl;
//    for(int i = 0; i<featuresMat.size(); i++) {
//        for (int j = 0; j < featuresMat.at(0).size(); j++)
//            cout << featuresMat[i][j] << " ";
//        cout<<"----->"<<targetValues[i];
//        cout<<endl;
//    }

}
