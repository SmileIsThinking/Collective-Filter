

// the initial letter of GPU function is in UPPER CASE
// the initial letter of CPU function is in LOWER CASE

// Global variable can not be used in kernel
#include <stdio.h>
#include <cuda_runtime.h>

#define Max 20000
// 80000 = 64 * 1250
#define testMax 2000
// 20000 ~= 64 * 320
#define Maxuser 2000

// search for the first position of every userId
__global__ void  Search(int* userId, int* userPos){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	if(i < 20000){
		j = userId[i];
		if(i > 0){
			if(userId[i] != userId[i-1]){
				userPos[j] = i;
			}
        }
	}
	__syncthreads();
}
// dont't get the sum in parallel right now
__global__ void Get_ravg(double* d_Ravg, int userNum,double* d_rating, int* d_userPos){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j;

	if(i < userNum+1){
		for(j = d_userPos[i]; j < d_userPos[i+1]; j++){
			d_Ravg[i] += d_rating[j];
		}
		d_Ravg[i] = d_Ravg[i]/(d_userPos[i+1] - d_userPos[i]);
	}
	__syncthreads();
}

__global__ void Get_ratingM(double* d_Ravg, int* d_userId, double* d_rating, double* d_ratingM, int k){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = 10000;
    if(i < j){
        d_ratingM[i] = d_rating[i] - d_Ravg[d_userId[i]];
//        d_ratingM[i] = 6.66;
    }
    __syncthreads();
}

int* get_userPos(int userNum,int* h_userPos, int* h_userId);
void read_training_set(int* h_userId, int* h_movieId, double* h_rating, int timestamp);
void read_test_set(int* h_testuserId, int* h_testmovieId, double* h_testrating, int timestamp);
double* get_ravg(int userNum, double* h_rating, int* d_userPos, double* h_Ravg);
double* get_ratingM(double* d_Ravg, int* h_userId, double* h_rating, double* h_ratingM);

int main(){
    // Read data from training set
	int h_userId[Max], h_movieId[Max] ,timestamp;
    double h_rating[Max];
    read_training_set(h_userId, h_movieId, h_rating, timestamp);

    // Read data from test set
    int h_testuserId[testMax], h_testmovieId[testMax];
    double h_testrating[testMax];
    read_test_set(h_testuserId, h_testmovieId, h_testrating, timestamp);


    // userPos is an array, storing the beginning position of the user.
    // Ex: userPos[2] = 9 means userId 2 starts from 9
    int userNum = h_userId[Max-1];
    int h_userPos[Maxuser] = {0};
    int *d_userPos;
    d_userPos = get_userPos(userNum, h_userPos, h_userId);
    for(int i = 1; i < userNum; i++){
        printf("%d\n userpos",h_userPos[i]);
//        printf("%d\n",h_userId[h_userPos[i]]);
//        printf("%d: %lf\n",h_userPos[i], h_Ravg[i]);
    }

    // get R average of every user
    double h_Ravg[Maxuser] = {0.0};
    double* d_Ravg;
    d_Ravg = get_ravg(userNum, h_rating, d_userPos, h_Ravg);


    // ratingm means that Rui - Ravg
    // After we get ratingm, we could figure out the Sim of i and j by scalar product (内积）
    // with sparse matrix library “cusparse”
    double h_ratingM[Max];
    double* d_ratingM;
    d_ratingM = get_ratingM(d_Ravg, h_userId, h_rating, h_ratingM);



    //just a test

    printf("userNum: %d\n",userNum);


     for(int i = 1; i < userNum; i++){
//         printf("%d:",h_userPos[i]);
//         printf("%d\n",h_userId[h_userPos[i]]);
         printf("userId: %d %d: %lf\n",h_userId[h_userPos[i]],h_userPos[i], h_Ravg[i]);
         printf("h_rating: %lf, d_ratingM: %lf\n", h_rating[i], h_ratingM[i]);
     }

    return 0;
}



// leave the ratingM in the global GPU memory
double* get_ratingM(double* d_Ravg, int* h_userId, double* h_rating, double* h_ratingM){
	int k;
	k = Max;

    int* d_userId;
    double* d_rating;
    double* d_ratingM;

    cudaMalloc((void **)&d_userId, Max);
    cudaMalloc((void **)&d_rating, Max);
    cudaMalloc((void **)&d_ratingM, Max);

    cudaMemcpy(d_rating, h_rating, Max, cudaMemcpyHostToDevice);
	cudaMemcpy(d_userId, h_userId, Max, cudaMemcpyHostToDevice);
    Get_ratingM<<<Max/128 + 1,128>>>(d_Ravg, d_userId, d_rating, d_ratingM, k);

    cudaMemcpy(h_ratingM, d_ratingM, Max, cudaMemcpyDeviceToHost);

    cudaFree(d_userId);
    cudaFree(d_rating);

    return d_ratingM;
}

// get R average of every user
double* get_ravg(int userNum, double* h_rating, int* d_userPos, double* h_Ravg){

    double* d_Ravg;
    double* d_rating;

    cudaMalloc((void **)&d_Ravg, userNum);
    cudaMalloc((void **)&d_rating, Max);

    cudaMemcpy(d_rating, h_rating, Max, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Ravg, h_Ravg, Max, cudaMemcpyHostToDevice);
    Get_ravg<<<10,128>>>(d_Ravg, userNum, d_rating, d_userPos);

    cudaMemcpy(h_Ravg, d_Ravg, userNum, cudaMemcpyDeviceToHost);
    cudaFree(d_rating);
    return d_Ravg;
}

// get userPos by GPU computation
int* get_userPos(int userNum,int* h_userPos, int* h_userId){
    // initialize
	h_userPos[userNum+1] = Max;
//    h_userPos[0] = -1;  // -1: does not exist

    // declare device variable and allocate device memroy
    int* d_userPos;
    int* d_userId;
    cudaMalloc((void **)&d_userPos, Maxuser);
    cudaMalloc((void **)&d_userId, Max);

    cudaMemcpy(d_userPos, h_userPos, Maxuser, cudaMemcpyHostToDevice);
    cudaMemcpy(d_userId, h_userId, Max, cudaMemcpyHostToDevice);

    Search<<<Max/128+1,128>>>(d_userId, d_userPos);

    cudaMemcpy(h_userPos, d_userPos, Maxuser, cudaMemcpyDeviceToHost);
    cudaFree(d_userId);
    return d_userPos;
}

// Read data from training set
void read_training_set(int* h_userId, int* h_movieId, double* h_rating, int timestamp){
    FILE *p = NULL;
    p = fopen("/usr/local/MovieLens/ml-100k/u1.base","r");
    if(p == NULL){
        printf("Read Error!\n");
        return;
    }
	for(int i = 0; i < Max; i++){
        fscanf(p,"%d %d %lf %d",&h_userId[i] , &h_movieId[i], &h_rating[i], &timestamp);
		// fscanf(p,"%d%*2c%d%*2c%lf%*2c%d",&h_userId[i] , &h_movieId[i], &h_rating[i], &timestamp);
//		printf("%d %d %lf\n",h_userId[i] , h_movieId[i], h_rating[i]);
	}
    fclose(p);
    return;
}

// Read data from test set
void read_test_set(int* h_testuserId, int* h_testmovieId, double* h_testrating, int timestamp){
    FILE *p = NULL;
    p = fopen("/usr/local/MovieLens/ml-100k/u1.test","r");
    if(p == NULL){
        printf("Read Error!\n");
        return;
    }
	for(int i = 0; i < testMax; i++){
        fscanf(p,"%d %d %lf %d",&h_testuserId[i] , &h_testmovieId[i], &h_testrating[i], &timestamp);
		// fscanf(p,"%d%*2c%d%*2c%lf%*2c%d",&h_userId[i] , &h_movieId[i], &h_rating[i], &timestamp);
		// printf("%d %d %lf\n",h_userId[i] , h_movieId[i], h_rating[i]);
	}
    fclose(p);
    return;
}

// get the MAE

//__global__ MAEcompute(double* pred, double* d_testrating, double* MAE){
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   double til;
//
//   if(pred[i] >= test[i]){
//       MAE[i] = pred[i] - d_testrating[i];
//   }else{
//       MAE[i] = d_testrating[i] - pred[i];
//   }
//   __syncthreads();
//}

//
//__device__ sumFunc(double number, int total){
//    i = blockIdx.x * blockDim.x + threadIdx.x;
//    int middle = total/2;
//    if()
//}


