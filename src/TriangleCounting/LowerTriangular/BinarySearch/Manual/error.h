#ifndef __error_h__ 
#define __error_h__ 

#include <cuda_runtime.h>
#include <stdio.h>
#include <string>

struct myErrorCuda {
    int devID;
    std::string fileName;
    int lineNumber;
    cudaError_t errorCode;
};

struct myError {
    std::string fileName;
    int lineNumber;
    std::string errorMessage;
};

#define ThrowCuda(STATEMENT) \
    do { cudaError_t e = cudaError::cudaSuccess; do { e = STATEMENT; } while (0); if (e != cudaError::cudaSuccess) { int devID = -1; cudaGetDevice(&devID); throw myErrorCuda{devID, std::string(__FILE__), __LINE__, e}; } } while (0);

#define Throw(MESSAGE) \
    do { throw myError{std::string(__FILE__), __LINE__, std::string((MESSAGE))}; } while (0);

#define TryCatch(STATEMENT) \
    try { \
        do { \
            STATEMENT; \
        } while (0); \
    } catch (myErrorCuda e) { \
        fprintf(stderr, \
            "[GPU%2d] %s:%d, %s(%d), %s\n", \
            e.devID, \
            e.fileName.c_str(), \
            e.lineNumber, \
            cudaGetErrorName(e.errorCode), \
            e.errorCode, \
            cudaGetErrorString(e.errorCode)); \
        cudaDeviceReset(); \
        exit(EXIT_FAILURE); \
    } catch (myError e) { \
        fprintf(stderr, \
            "%s:%d, %s\n", \
            e.fileName.c_str(), \
            e.lineNumber, \
            e.errorMessage.c_str()); \
        exit(EXIT_FAILURE); \
    }

#endif