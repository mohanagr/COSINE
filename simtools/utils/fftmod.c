#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <complex.h>
#include <fftw3.h>
#define FFTW_FLAG FFTW_ESTIMATE
//module load intel/2019u3  openmpi/4.0.1 fftw/3.3.10 python/3.8.5
//icc -std=c99 -O3 -xHost -shared -o libpfb.so -fPIC -fopenmp fftmod.c -lfftw3_omp -lm
//gcc -std=c99 -O3 -march=native -shared -o libpfb.so -fPIC -fopenmp fftmod.c -lfftw3_omp -lm
// void myfft(complex *input, complex *output, int64_t n)
// {   
//     if(fftw_init_threads()) {
//         fftw_plan p;
//         fftw_plan_with_nthreads(8);
//         p = fftw_plan_dft_1d(n, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
//         fftw_execute(p);
//         fftw_destroy_plan(p);
//         fftw_cleanup_threads();
//     }
//     else {
//         printf("something went wrong, sorry");
//     }
// }

int set_threads(int nthreads)
{
  if (nthreads<0) {
    nthreads=omp_get_num_threads();
  }
  if(fftw_init_threads()){
    fftw_plan_with_nthreads(nthreads);
    printf("Set FFTW to have %d threads.\n",nthreads);
  }
  else{
    printf("something went wrong during thread init");
  }
}

void fft_c2r_1d(fftw_complex *datft,double *dat, int64_t n, int preserve_input)
{
  fftw_plan plan;
  if(preserve_input){
    plan=fftw_plan_dft_c2r_1d(n,datft,dat,FFTW_FLAG|FFTW_PRESERVE_INPUT);
  }
  else{
    plan=fftw_plan_dft_c2r_1d(n,datft,dat,FFTW_FLAG|FFTW_PRESERVE_INPUT);
  }
  fftw_execute(plan);
  fftw_destroy_plan(plan);
  //apply the normalization so the inverse of the forward gives you what you started with
  double nn=1.0/n;
  #pragma omp parallel for
  for (int64_t i=0;i<n;i++)
    dat[i]*=nn;
  
}

void fft_r2c_1d(double *dat, fftw_complex *datft, int64_t n)
{
  fftw_plan plan=fftw_plan_dft_r2c_1d(n, dat, datft, FFTW_FLAG);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
}

void test(double* input, double* output, int64_t nrows, int64_t ncols){
    //  for(int i =0; i<n;i++){
    //      printf("r %f im %f \n",creal(input[i]),cimag(input[i]));
    //  }
    // complex *cp;
    // cp = (complex *) input;
    int n[] = {4};
    fftw_plan p;
    // const int inembed[] = {6};
    // const int oembed[] = {4};
    
    printf("nrows %d ncols %d", nrows, ncols);
     for(int i =0; i<nrows;i++){
      for(int j =0; j<ncols; j++)
      {
        output[i*(ncols+2)+j] = input[i*ncols+j];
      }
        //  printf("r %f im %f \n",creal(cp[i]),cimag(cp[i]));
        
     }
    // for(int i =0; i<n;i++){
    //      printf("r %f im %f \n",input[2*i],input[2*i+1]);
    //  }
    for(int i=0; i<nrows;i++){
      printf(" \n");
      for(int j=0; j<ncols; j++)
      {
        printf("%d,%d %f ",i,j, output[i*ncols+j]);
      }
        //  printf("r %f im %f \n",creal(cp[i]),cimag(cp[i]));
        
     }
    // for(int i =0; i<n;i++){
    //      output[2*i]= i;
    //      output[2*i+1]=i;
    //  }
    
    // p = fftw_plan_dft_r2c_1d(n, output, (fftw_complex*)output, FFTW_ESTIMATE);
    p=fftw_plan_many_dft_r2c(1,n,2,output,NULL,1,6,(fftw_complex *)output,NULL,1,3,FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
}

void pfb(double *timestream, double *spectra, double *window, const int64_t nspec, const int64_t nchan, const int64_t ntap){
    int64_t lblock = 2*nchan;
    // double *pseudo_ts;
    // pseudo_ts = (double *)malloc(sizeof(double)*lblock*nspec);
    #pragma omp parallel for
    for(int i=0; i<nspec; i++){
        for(int j=0; j<lblock; j++){
            // printf("going to access location (%d,%d)\n", i,j);
            spectra[i*(lblock+2) + j]=0;
            for(int k=0; k<ntap; k++){
                // printf("going to add %f to location\n", timestream[(i+k)*lblock+j]*window[k*lblock+j]);
                spectra[i*(lblock+2) + j] += timestream[(i+k)*lblock+j]*window[k*lblock+j];
            }
        }
    }
        fftw_plan p;
        int n[] = {lblock}; /* 1d transforms of length 10 */
        p = fftw_plan_many_dft_r2c(1, n, nspec, spectra, NULL, 1, lblock+2, (fftw_complex *)spectra, NULL, 1, nchan+1, FFTW_ESTIMATE);
        fftw_execute(p);
        fftw_destroy_plan(p);
}

void cleanup_threads(){
  fftw_cleanup_threads();
}
