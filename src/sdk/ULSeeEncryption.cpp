//
// Created by kun on 17-3-8.
//

#include "ULSeeEncryption.h"
#include <fstream>
#include <algorithm>
#include <map>

using namespace std;

ULSeeEncryption::ULSeeEncryption()
{
    charToByte(bkey, key.c_str());

    KeyExpansion(bkey, wkey);
}

ULSeeEncryption::ULSeeEncryption(const vector<string> &fileLists) : fileNames(fileLists) {

    charToByte(bkey, key.c_str());

    KeyExpansion(bkey, wkey);
}


string ULSeeEncryption::produceHead() {
    string head = this->prefix;
    head += "aaabbbcccdddeeefff"; //Add the padding
    return head;
}


/*
 * First try not use the zlip library, using C_Area after the header, and do des encryption on the C_Area
 * */
bool ULSeeEncryption::doEncryption(const string &outFile) {
    string CArea;
    //finish the output stream fuse
    ofstream output(outFile, ios_base::binary);
    string head = produceHead();


    vector<char> datas;
    for (auto filename: fileNames) {
        vector<char> data;
        CArea += filename;
        ifstream ifs(filename, std::ifstream::ate | std::ifstream::binary);
        int len = ifs.tellg();
        ifs.seekg(0, std::ios_base::beg);
        data.resize(len);
        CArea += "___" + to_string(len) + "___";
        ifs.read(&data[0], len);
        ifs.close();
        datas.insert(datas.end(), data.begin(), data.end());
    }
    CArea = CArea.substr(0, CArea.size() - 3); // erase the last "___"
    string tails = AESInd + CArea;
    head = "_" + to_string(tails.length());


    //parse the index and do the encryption according to the AESInd
    int i = 0;
    // 2---len-2, erase the length of "__"
    string indexS;


    for (i = 2; i < AESInd.size() - 1; i++) {
        if (AESInd[i] != '_') {
            indexS += AESInd.at(i);
        } else {
            int index = stoi(indexS);
		    //int index;
			//std::stoi(indexS,index,10);
            if (index > datas.size() - 129)
                break;
            indexS.clear();
            int x, j;
            byte einfo[16];
            for (x = 0, j = index; x < 16; x++)
                einfo[x] = byte(datas[j++]);
            encrypt(einfo, wkey);
            for (x = 0; x < 16; x++) {
                datas[index + x] = static_cast<char>(einfo[x].to_ulong());
            }
            //encrypt((byte*)(&datas[0] + index), wkey);
        }
    }

    //output.seekp(output.end);

    //encryption the AESInd and fileName info(CArea)

    byte tailsBytes[tails.size()];
    cout << "Before encryption, tails is :" << tails << endl;
    encryptString(tails, bkey, tailsBytes);
    cout << "After encryption, tails is :" << tails << endl;

    auto res = crackString(tailsBytes, bkey, tails.size());

    for (int j = 0; j < tails.size(); j++) {
        datas.push_back((char) (tailsBytes[j].to_ulong()));
    };
    for (int x = 0; x < head.size(); x++) {
        datas.push_back(head[x]);
    }

    output.write(&datas[0], datas.size());
    output.flush();
    output.close();
    return true;
}

/*
 *
 * Try do decode from the encryption file
 * */
/*
 *
 * Try do decode from the encryption file
 * */
map<string, vector<char> > ULSeeEncryption::doDecoder(const string &filename, string wdiskFolder) {
    vector<char> datas;
    ifstream ifs(filename, std::ifstream::ate | std::ifstream::binary);
    int len = ifs.tellg();
    ifs.seekg(0, std::ios_base::beg);
    datas.resize(len);
    ifs.read(&datas[0], len);
    //first we decode the header

    //todo equal to "ULSee_EncrYpTioN_Ver1.0"
    //todo omit the padding "aaabbbcccdddeeefff"
    string nums;
    int be = datas.size() - 1;
    while (datas[be] != '_') {
        nums += datas[be--];
    }
    reverse(nums.begin(), nums.end());
    int inum = stoi(nums);
    //be--;

    //second we decode the AESInd and fileName info(CArea)
    byte encryptionInfo[inum];
    for (int i = be - inum, j = 0; j < inum; j++, i++) {
        encryptionInfo[j] = byte(datas[i]);
    }
    //string cInfo = crackString((byte*)(&datas[0] + (be - inum)), bkey, inum);
    string cInfo = crackString(encryptionInfo, bkey, inum);
    //cout << "cInfo is :" << cInfo << endl;

    //now we docode the cInfo to get the file list and the encryption index.

    vector<int> aesInd, fileSize;
    vector<string> filenames;

    // a: parse the encrypted index
    string d = "__";
    auto snum = cInfo.find(d);
    string num;
    for (int j = 2; j < snum + 1; j++) {
        if (cInfo[j] != '_')
            num += cInfo[j];
        else {
            aesInd.push_back(stoi(num));
            num.clear();
        }
    }

    // b: parse the file list and file size
    string fInfo = cInfo.substr(snum + 2);
    string delimiter = "___";

    size_t pos = 0;
    int i = 1, totalsize = 0;
    while ((pos = fInfo.find(delimiter)) != std::string::npos) {
        string token = fInfo.substr(0, pos);
        fInfo.erase(0, pos + delimiter.length());
        if (i % 2 == 1)
            filenames.push_back(token);
        else {
            totalsize += stoi(token);
            fileSize.push_back(stoi(token));
        }
        i++;
    }
    totalsize += stoi(fInfo);
    fileSize.push_back(stoi(fInfo));


    //decode the encryption index of the binary file
    for (auto index : aesInd) {
        if (index > totalsize - 129)
            break;
        byte einfo[16];
        int x, j;
        for (x = 0, j = index; x < 16; x++)
            einfo[x] = byte(datas[j++]);
        decrypt(einfo, wkey);
        for (x = 0; x < 16; x++) {
            datas[index + x] = static_cast<char>(einfo[x].to_ulong());
        }
        //decrypt((byte*)(&datas[0] + index), wkey);
    }

    //now we segment the whole file to some little files
    map<string, vector<char>> originFiles;
    int ind = 0;
    for (int r = 0; r < filenames.size(); r++) {
        //vector<char>(datas.begin() + ind, datas.begin() + ind + fileSize[r]))
        vector<char> data(datas.begin() + ind, datas.begin() + ind + fileSize[r]);
        originFiles.insert(make_pair(filenames[r], data));
        ind += fileSize[r];
    }

    //write decode files to disk
    if (!wdiskFolder.empty()) {
        for (auto ele : originFiles) {
            string filepath = wdiskFolder + "/" + ele.first;
            ofstream fout(filepath, ios::out | ios::binary);
            fout.write(&(ele.second[0]), ele.second.size());
            fout.close();
        }
    }


    // cout << fInfo;
    return originFiles;
}

