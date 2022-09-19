#include <stdio.h>
#include <omp.h>


void *get_dynamic_shared() { return NULL; }
#pragma omp begin declare variant match(device = {arch(nvptx64)})
extern void *__kmpc_get_dynamic_shared(void);
void *get_dynamic_shared() { return __kmpc_get_dynamic_shared(); }
#pragma omp end declare variant

uint32_t get_thread_id() {return -1;}
#pragma omp begin declare variant match(device = {arch(nvptx64)})
extern uint32_t __kmpc_get_hardware_thread_id_in_block(void);
uint32_t get_thread_id() { return __kmpc_get_hardware_thread_id_in_block(); }
#pragma omp end declare variant

uint32_t get_barrier() {return -1;}
#pragma omp begin declare variant match(device = {arch(nvptx64)})
extern uint32_t  __kmpc_barrier_simple_spmd(int *Loc, int32_t TId); ;
uint32_t get_barrier() { return __kmpc_barrier_simple_spmd(NULL,NULL); }
#pragma omp end declare variant


#pragma omp begin declare target device_type(nohost)
int BaseLoc[2];
#pragma omp allocate(BaseLoc) allocator(omp_pteam_mem_alloc)   

int GlobalBase;
#pragma omp allocate(GlobalBase) allocator(omp_pteam_mem_alloc)  

int *DataBuf;
#pragma omp allocate(DataBuf) allocator(omp_pteam_mem_alloc) 

int LoadCounter;
#pragma omp allocate(LoadCounter) allocator(omp_pteam_mem_alloc)  

#pragma omp end declare target


#pragma omp begin declare target 

long int padding_func(long int num)
{
    if(num % 8 == 0)
        return num+1;
    else
        return num;
}

int *shared_copy_thread()
{
    int *DataBuf = (int*) get_dynamic_shared();
    return DataBuf;
}

int shared_copy_team(long int sharedIv, long int teamLb, int *V, long int Base, long int Step, long int num)
{
    
    int Tid = get_thread_id();
    int *DataBuf = (int*) get_dynamic_shared(); 
    int b = Base+Tid*Step;
    int itr = sharedIv-teamLb;
    int bufOff;
    if(num % 8 != 0)
        bufOff = (sharedIv-teamLb)*num + Tid;
    else
        bufOff = (sharedIv-teamLb)*(num+1) + Tid;
    int iPlus = omp_get_max_threads();

    if((itr) !=0 && Base == GlobalBase)
        return 1;

    if(Tid == 0)
    {
        if(sharedIv-teamLb == 0)
            BaseLoc[0] = 0;
        else
        {
            int c = (Base != GlobalBase);
            BaseLoc[sharedIv-teamLb] = c + BaseLoc[sharedIv-teamLb-1] ; 
        }
        GlobalBase = Base;
    }

    int bufOffmix = BaseLoc[itr]*num + Tid; 

    for(int i=0 ; i<num ; i+=iPlus)
        DataBuf[bufOffmix+i] = V[b + i*Step];
    
    return 1;
}

#pragma omp end declare target

