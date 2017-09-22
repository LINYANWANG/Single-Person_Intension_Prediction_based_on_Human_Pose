//
// Created by kun on 17-3-8.
//

#ifndef ENCRYPTION_ULSEEENCRYPTION_H
#define ENCRYPTION_ULSEEENCRYPTION_H

#include <string>
#include <vector>
#include "aes.h"
#include <map>
using namespace std;




class ULSeeEncryption {
public:
    ULSeeEncryption();
    ULSeeEncryption(const vector<string>& fileLists);
    bool doEncryption(const string& filename);
    map<string, vector<char>>  doDecoder(const string& filename, string wdiskFolder = "decode");
    //temp public todo private
    byte bkey[16];

private:
    string produceHead();
    string prefix = "ULSee_EncrYpTioN_Ver1.0";
    int padNum = 18; // bit
    string AESInd = "000_200_40000_60000_80000__"; // and the last 128bit;
    vector <string>  fileNames; // the files name for encryption;
    string key = "abcdefghijklmnop";
    word wkey[4*(Nr+1)];
};


#endif //ENCRYPTION_ULSEEENCRYPTION_H
