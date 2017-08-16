#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <string.h>
#include <stdlib.h>
using namespace std;
enum DataType {INT, FLOAT, STRING};

struct SchemaTuple {
    string name;
    DataType type;
    bool isResult;
    SchemaTuple(string name, DataType type, bool isResult) : name(name), type(type), isResult(isResult) {};
};


void readCsv(char * filename, float * ans, int &numSamples, int &numAttrs);
void splitString(vector<float> &line, char * str);
void splitString1(vector<float> &line, char * str);

void readCsv(char * filename, float * ans, int &numSamples, int &numAttrs) {
    ifstream file(filename);
    char value[10000];
    int i = 0;
    int offset = 0;
    while ( file.good() )
    {
        if( i >= 50000 ) {
            file.close();
            numSamples = i;
            return;
        }
        getline(file, value);
        //cout<<value<<endl;
        cout<<i<<endl;
        vector<float> parts;
        splitString1(parts, value);
        for(int k = 0; k<parts.size(); k++) {
            float val = parts[k];
            memcpy(ans+offset, &val, sizeof(float));
            offset++;
        }
        if(i == 0)
            numAttrs = offset;
        i++;
    }
    file.close();
}

/*
char *substring(char *string, int position, int length) 
{
   char *pointer;
   int c;
 
   pointer = malloc(length+1);
 
   if (pointer == NULL)
   {
      printf("Unable to allocate memory.\n");
      exit(EXIT_FAILURE);
   }
 
   for (c = 0 ; c < position -1 ; c++) 
      string++; 
 
   for (c = 0 ; c < length ; c++)
   {
      *(pointer+c) = *string;      
      string++;   
   }
 
   *(pointer+c) = '\0';
 
   return pointer;
}


void splitString(vector<float> &line, char * str) {
    int start = 0;
    int len = sizeof(str);
    for(int i = 0; i<len; i++) {
        if(str[i]==';' || str[i]==','){
            line.push_back(atoi(substring(start, i-start)));
            start = i+1;
        }
    }
    line.push_back(atoi(substring(start,len-start)));
}
*/
void splitString1(vector<float> &line, char * str) {
    char * pch;
    pch = strtok (str,",");
    while (pch != NULL)
    {
        printf ("%s\n",pch);
        line.push_back(atoi(pch));
        pch = strtok (NULL, ",");
    }
}

void createSchema(vector<SchemaTuple*>& schema) {
    schema.push_back(new SchemaTuple("sepal_l", FLOAT, false));
    schema.push_back(new SchemaTuple("sepal_w", FLOAT, false));
    schema.push_back(new SchemaTuple("petal_l", FLOAT, false));
    schema.push_back(new SchemaTuple("petal_w", FLOAT, false));
    schema.push_back(new SchemaTuple("class", STRING, true));
}
