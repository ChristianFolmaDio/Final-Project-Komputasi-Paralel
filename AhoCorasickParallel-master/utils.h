#ifndef UTILS_H__
#define UTILS_H__

#ifndef __cplusplus
typedef char bool;
#endif

#include <iostream>
#include <iterator>
#include <fstream> 
#include <sstream>
#include <vector>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <algorithm>
#ifdef _WIN32
// Windows tidak memiliki unistd.h, jadi kita definisikan fungsi yang diperlukan atau kosongkan
#define sleep(x) Sleep(1000 * (x))
#else
#include <unistd.h>
#endif
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#include <time.h>
#else
#include <sys/time.h>
#endif
#define true (char)1
#define false (char)0

#define max_dna_length 1000  // Changed from max_tweet_length

#endif