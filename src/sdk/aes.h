//
// Created by kun on 17-3-9.
//

#ifndef ENCRYPTION_AES_H
#define ENCRYPTION_AES_H


#include <iostream>
#include <bitset>
#include <string>

using namespace std;
typedef bitset<8> byte;
typedef bitset<32> word;

const int Nr = 10;  // AES-128需要 10 轮加密
const int Nk = 4;   // Nk 表示输入密钥的 word 个数

void encrypt(byte in[4 * 4], word w[4 * (Nr + 1)]);

void decrypt(byte in[4 * 4], word w[4 * (Nr + 1)]);

void charToByte(byte out[16], const char s[16]);

void divideToByte(byte out[16], bitset<128> &data);

void KeyExpansion(byte key[4 * Nk], word w[4 * (Nr + 1)]);

void encryptString(const string &in, byte key[16], byte *out);

string crackString(byte *in, byte key[16], int len);

#endif //ENCRYPTION_AES_H
