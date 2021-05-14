/* COMP.CE.350 Parallelization Excercise 2020
   Copyright (c) 2016 Matias Koskela matias.koskela@tut.fi
                      Heikki Kultala heikki.kultala@tut.fi

VERSION 1.1 - updated to not have stuck satellites so easily
VERSION 1.2 - updated to not have stuck satellites hopefully at all.
VERSION 19.0 - make all satelites affect the color with weighted average.
               add physic correctness check.
VERSION 20.0 - relax physic correctness check
*/

// Example compilation on linux
// no optimization:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm
// most optimizations: gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2
// +vectorization +vectorize-infos: gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec
// +math relaxation:  gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math
// prev and OpenMP:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math -fopenmp
// prev and OpenCL:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math -fopenmp -lOpenCL

// Example compilation on macos X
// no optimization:   gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL
// most optimization: gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL -O3



#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h> // printf
#include <math.h> // INFINITY
#include <stdlib.h>
#include <string.h>
#include <assert.h>
// Window handling includes
#ifndef __APPLE__
#include <GL/gl.h>
#include <GL/glut.h>
#else
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#endif



// OpenCL includes
#include <CL/cl.h>

// include GL libabry
#include <GL/freeglut.h>

//==================================================This will be included in header file============================

// These are used to decide the window size
#define WINDOW_HEIGHT 1024
#define WINDOW_WIDTH  1024

// The number of satelites can be changed to see how it affects performance.
// Benchmarks must be run with the original number of satellites
#define SATELITE_COUNT 64

// These are used to control the satelite movement
#define SATELITE_RADIUS 3.16f
#define MAX_VELOCITY 0.1f
#define GRAVITY 1.0f
#define DELTATIME 32
#define PHYSICSUPDATESPERFRAME 100000

// Some helpers to window size variables
#define SIZE WINDOW_HEIGHT*WINDOW_HEIGHT
#define HORIZONTAL_CENTER (WINDOW_WIDTH / 2)
#define VERTICAL_CENTER (WINDOW_HEIGHT / 2)
// Stores 2D data like the coordinates
typedef struct{
   float x;
   float y;
} floatvector;

// Stores 2D data like the coordinates
typedef struct{
   double x;
   double y;
} doublevector;

// Stores rendered colors. Each float may vary from 0.0f ... 1.0f
typedef struct{
   float red;
   float green;
   float blue;
} color;

// Stores the satelite data, which fly around black hole in the space
typedef struct{
   color identifier;
   floatvector position;
   floatvector velocity;
} satelite;

//============================//============================//============================//============================	


// Is used to find out frame times
int previousFrameTimeSinceStart = 0;
int previousFinishTime = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;





// Pixel buffer which is rendered to the screen
color* pixels;

// Pixel buffer which is used for error checking
color* correctPixels;

// Buffer for all satelites in the space
satelite* satelites;
satelite* backupSatelites;



// Add my own additional variable
// Is used to find out frame times

//#define WG_SIZE_X 8
//#define WG_SIZE_Y 8

#define PROGRAM_SOURCE "parallelGraphicEngine.cl"
#define KERNEL_FUNCTION "parallelGraphics"

#define TOTAL_PIXEL_SIZE sizeof(color) * SIZE
#define TOTAL_SATELLITE_SIZE sizeof(satelite) * SATELITE_COUNT
#define MAX_SOURCE_SIZE (0x100000)



// ## You may add your own variables here ##
/*
typedef struct __attribute__((packed)) _satelite{
	color identifier;
	cl_floatvct position;
	cl_floatvct velocity;
	
}satelite;

typedef struct __attribute__((packed)) _color{
	cl_float red;
	cl_float green;
	cl_float blue;
} color;

typedef struct __attribute__ ((packed)) _cl_doublevct{
   cl_float x;
   cl_float y;
} cl_doubletvct;

typedef struct __attribute__ ((packed)) _cl_floatvct{
   cl_float x;
   cl_float y;
} cl_floatvct;
*/



/*
cl_command_queue Physics_cmd_Queue;
cl_command_queue Graphics_cmd_Queue;


cl_context Physics_context;
cl_context Graphics_context;


cl_mem Physics_bufSat;
cl_mem Graphics_bufSat;
cl_mem bufPix;


cl_device_id devices;

cl_program Physics_program;
cl_program Graphics_program;

cl_kernel Physics_kernel;
cl_kernel Graphics_kernel;
*/


cl_int status;
cl_event kernel_events;
cl_command_queue physics_cmd_queue = NULL;
cl_kernel physics_kernel = NULL;
cl_command_queue graphics_cmd_queue = NULL;
cl_mem graphics_satelites_buff = NULL;
cl_kernel graphics_kernel = NULL;
cl_mem pixels_buff = NULL;
cl_mem physics_satelites_buff = NULL;
cl_program physics_program = NULL;
cl_program graphics_program = NULL;
cl_context physics_context = NULL;
cl_context graphic_context = NULL;
  
size_t local_size[2];
const char* option = "-I parallel.h -cl-fast-relaxed-math";

void set_local_size(){
   printf("Enter the WG x coordinate frame size: ");
   assert(scanf("%zu", &local_size[0]) > 0);
   printf("Enter the WG y coordinate frame size: ");
   assert(scanf("%zu", &local_size[1]) > 0);
}


cl_device_id GetDeviceIDs(cl_device_type device_type){

  	printf("Start GetDeviceID function () \n");
	cl_platform_id  *platforms ;
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint numPlatforms;
	cl_uint numDevices;
	//cl_int status;


	//Get number of platforms
	status = clGetPlatformIDs(0,NULL,&numPlatforms);
	assert(status == CL_SUCCESS);

	//Obtain platform using function malloc
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatforms);
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	assert(status == CL_SUCCESS);

	/*
	printf("%d\n",status);
	if (status < 0){
		perror("Could not identify the platform");

	}
	*/
	

	for (int i = 0 ; i < numPlatforms; i++){
	  status = clGetDeviceIDs(platforms[i], device_type, 0, NULL, &numDevices);
	  cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id)*numDevices); 
	  clGetDeviceIDs(platforms[i],device_type,numDevices,devices,NULL);
	
	
	/*
	printf("%d\n",status);
	if (status == CL_DEVICE_NOT_FOUND){
		perror("Could not find the CPU");
	
		}
	*/
		if (status == CL_SUCCESS){
	  		return devices[0];
		}else{
	  		continue;
		}

	}	
	printf("End GetDeviceID function\n ()");
}	


/*
cl_program build_program(cl_context context,cl_device_id devices, const char* source ){
	
	
	FILE* program_handle;
	char* program_buffer;
	
	size_t program_size;
	int status;
	
	/*fopen_s(&program_handle, source, "rb");
	if (!program_handle){
	
		perror("Fail to load program/kernel file");
		exit(1);
	}
	//fseek(program_handle, 0, SEEK_END);
	//program_size = ftell(program_handle);
	//rewind(program_handle);
	//program_buffer = (char*)malloc(program_size+1);
	//program_buffer[program_size] = '\0';
	//fread(program_buffer,sizeof(char), program_size, program_handle);
	//fclose(program_handle);
	
	cl_program program;
	program = clCreateProgramWithSource(context, 1, (const char**)program_buffer,&program_size, &status);
	printf("%d\n",status);
	if (status < 0){
	
		perror("Could not create the program");
		//exit(1);
	}
	
	
	char* program_log;
	size_t log_size;
	 // Build (compile) the program for the device
	status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	printf("%d\n",status);
	if (status < 0){
	
		clGetProgramBuildInfo(program, devices, CL_PROGRAM_BUILD_LOG,0,NULL,&log_size);	
		program_log = (char*)malloc(log_size + 1);		
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, devices, CL_PROGRAM_BUILD_LOG, log_size+1, program_log,NULL);
		printf("%s\n",program_log);
		//exit(1);	
	}
	
	return program;
		

}
*/


void set_physics_engine(char *source_str, size_t source_size){
  printf("Start set_physics_engine funtion ()\n");
  //CPU will execute the physics engine
  cl_device_id gpu_id = GetDeviceIDs(CL_DEVICE_TYPE_GPU);
  //printf("End GetDeviceID function in set_physics_engine\n");
  //printf("%d\n",cpu_id);

  //======================================================
/*
std::vector<Device> devices;
    std::vector<Platform> platforms; // get all platforms
    std::vector<Device> devices_available;
    int n = 0; // number of available devices
    Platform::get(&platforms);
    for(int i=0; i<(int)platforms.size(); i++) {
        devices_available.clear();
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices_available);
        if(devices_available.size()==0) continue; // no device found in plattform i
        for(int j=0; j<(int)devices_available.size(); j++) {
            n++;
            devices.push_back(devices_available[j]);
        }
    }
    if(platforms.size()==0||devices.size()==0) {
        std::cout << "Error: There are no OpenCL devices available!" << std::endl;
        return -1;
    }
    for(int i=0; i<n; i++) std::cout << "ID: " << i << ", Device: " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
}*/
//======================================================

  //Create a context for Physics Engine
  physics_context = clCreateContext(NULL,1,&gpu_id,NULL,NULL,&status);
  printf("1_");
  printf("%d\n",status);
  assert(status == CL_SUCCESS);

  //Create a cmd queue for Physics Engine
  
  physics_cmd_queue = clCreateCommandQueue(physics_context, gpu_id, 0 , &status);
  printf("2_");
  printf("%d\n",status);
  assert(status == CL_SUCCESS);

  //Create a buffer for holding satelites data in Physics Engine

  physics_satelites_buff = clCreateBuffer(physics_context,CL_MEM_USE_HOST_PTR,TOTAL_SATELLITE_SIZE,satelites,&status);
  printf("3_");
  printf("%d\n",status);
  clFinish(physics_cmd_queue);


  //Create a program from source string

  physics_program = clCreateProgramWithSource(physics_context, 1, (const char**)&source_str, (const size_t *)&source_size, &status);  
  printf("4_");
  printf("%d\n",status);
  assert(status == CL_SUCCESS);

	
  //Build program and print error if it exists
  //"-cl-fast-relaxed-math"
  status = clBuildProgram(physics_program,1,&gpu_id,"-cl-fast-relaxed-math",NULL,NULL); 
  printf("5_");
  printf("%d\n",status);
 	
  if (status != CL_SUCCESS){
    if (status == CL_BUILD_PROGRAM_FAILURE) {
    printf("Build program not success\n");
    //cl_int status_code;
    size_t log_len;

    clGetProgramBuildInfo(physics_program, cpu_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
    //printf("status_code %d",status_code);

    char* buff_err = malloc(log_len);
    clGetProgramBuildInfo(physics_program, cpu_id, CL_PROGRAM_BUILD_LOG, log_len, buff_err, NULL);
    
   
    //Print the log
    printf("%s\n",buff_err);
    
	
    /*	
    if (status_code){
      printf("clGetProgramBuildInfo fail at line %d\n", status_code);
      exit(-1);
    }

    buff_err = malloc(log_len);
    if (!buff_err){
      printf("Malloc failed at line %d\n", __LINE__);
      exit(-2);
    }

    status_code = clGetProgramBuildInfo(physics_program, cpu_id, CL_PROGRAM_BUILD_LOG, log_len, buff_err, NULL);
    if (status_code){
      printf("clGetProgramBuildInfo fail at line %d\n", __LINE__);
      exit(-3);
    }*/


  // Free resource
  free(buff_err);

  printf("6_EXIT FAILURE_");
  printf("%d\n",status);
  // Exit failure
  exit(EXIT_FAILURE);
  }
  }

  //Start to create a physics engine
  printf("Start to create a physics engine kernel");
	
  physics_kernel = clCreateKernel(physics_program, "parallelPhysicsEngineKernel",&status);
  //printf("5_");
  //printf("%d\n",status);
  
  status = clSetKernelArg(physics_kernel, 0, sizeof(cl_mem), (void*)&physics_satelites_buff);
  
  //printf("6_");
  //printf("%d\n",status);
  
  assert(status == CL_SUCCESS);

  clFinish(physics_cmd_queue);
  printf("Finish set_physics_engine funtion ()\n");
}



void set_graphics_engine(char* source_str, size_t source_size){

  printf("Start set_graphics_engine funtion ()\n");
  //GPU will execute the graphic loop
  cl_device_id gpu_id = GetDeviceIDs(CL_DEVICE_TYPE_GPU);

  //Create a context for graphic engine
 
  graphic_context = clCreateContext(NULL,1, &gpu_id, NULL, NULL , &status);
  assert(status == CL_SUCCESS);

  // Create command queue for Graphics Engine

  graphics_cmd_queue = clCreateCommandQueue(graphic_context, gpu_id, 0, &status);
  assert(status == CL_SUCCESS);
  
  //// Create a buffer that will filled in with satelites' dataCreate a buffer for Graphic Engine

  graphics_satelites_buff = clCreateBuffer(graphic_context, CL_MEM_USE_HOST_PTR, TOTAL_SATELLITE_SIZE,satelites, &status);
  assert(status == CL_SUCCESS);

  ////Create a buffer that will filled in with pixels ' data for Graphic Engine

  pixels_buff = clCreateBuffer(graphic_context, CL_MEM_USE_HOST_PTR, TOTAL_SATELLITE_SIZE, pixels, &status);
  assert (status == CL_SUCCESS);

  clFinish(graphics_cmd_queue);
  
  //Create a program from source code (in string) in Graphic Engine loop

  graphics_program = clCreateProgramWithSource(graphic_context, 1, (const char**)&source_str, (const size_t *)&source_size, &status);
  assert(status == CL_SUCCESS);

  //Build program and print error if it exists (similar to prior implementaion in physics engine loop)
  status = clBuildProgram(graphics_program,1,&gpu_id, option, NULL,NULL);
  if (status != CL_SUCCESS){
    
    if (status == CL_BUILD_PROGRAM_FAILURE) {
    printf("Build program not success\n");
    //cl_int status_code;
    size_t log_len;

    clGetProgramBuildInfo(physics_program, gpu_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
    //printf("status_code %d",status_code);

    char* buff_err = malloc(log_len);
    clGetProgramBuildInfo(physics_program, gpu_id, CL_PROGRAM_BUILD_LOG, log_len, buff_err, NULL);
    
   
    //Print the log
    printf("%s\n",buff_err);

    /*
    clGetProgramBuildInfo(graphics_program, gpu_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);

    if (status_code){
      printf("clGetProgramBuildInfo fail at line %d\n", __LINE__);
      exit(-1);
    }

    buff_err = malloc(log_len);
    if (!buff_err){
      printf("Malloc failed at line %d\n", __LINE__);
      exit(-2);
    }

    status_code = clGetProgramBuildInfo(graphics_program, gpu_id, CL_PROGRAM_BUILD_LOG, log_len, buff_err, NULL);
    if (status_code){
      printf("clGetProgramBuildInfo fail at line %d\n", __LINE__);
      exit(-3);
    }*/


    // Free resource
    free(buff_err);
  
    // Exit failure
    exit(EXIT_FAILURE);
    }
  }

  //Start to create a graphics kernel
  printf("Start to create a graphics engine kernel");
  graphics_kernel = clCreateKernel(graphics_program, "parallelGraphicsEngineKernel",&status);
  
  status = clSetKernelArg(graphics_kernel, 0, sizeof(cl_mem), (void*)&graphics_satelites_buff);
  status = clSetKernelArg(graphics_kernel, 1, sizeof(cl_mem), (void*)&pixels_buff);
  assert(status == CL_SUCCESS);

  clFinish(graphics_cmd_queue);
  printf("Finish set_graphics_engine funtion ()\n");

}



// ## You may add your own initialization routines here ##
void init(){
  printf("Start init function () \n");
  FILE *file;

  char *source_string;
  size_t source_size;
  printf("Start open file cl \n");		
  file = fopen("cl_parallelGraphicsEngine.cl","r");
  if (!file){
    printf("Fail to load the kernel file\n");
    exit(1);
  }

  source_string = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_string,1,MAX_SOURCE_SIZE,file);
  fclose(file);
  printf("Finish open file cl \n");	

  //Set the physics engines
  printf("Start call set up physics engine in init\n");
  set_physics_engine(source_string,source_size);
  printf("Finish call set up physics engine in init\n");

  //Set the graphics engines
  printf("Start call set up graphics engine in init\n");
  set_graphics_engine(source_string,source_size);
  printf("Finish call set up graphics engine in init\n");

  //Set up WG size
  printf("Start call set_local_size\n");
  set_local_size();
  printf("Finish call set_local_size\n");

  printf("Finish init function () \n");
}



  /*
	devices = create_device();
	cl_context context;
	context = clCreateContext(NULL,1,&devices, NULL,NULL,&status);
	printf("%d\n",status);
	if(status < 0){
		
		perror("Could not create a context");
		
		//exit(1);
	}
	
	
	
	program = build_program(context, devices, PROGRAM_SOURCE );
	
	 // Create the vector additis<on kernel
    	cl_kernel kernel;
    	kernel = clCreateKernel(program, "parallelGraphicEngine", &status);
    	printf("%d\n",status);
    	if (status < 0){
    		
    		perror("Could not create a kernel");
    		//exit(1);
    	}
    	
	
	// Create a buffer that will filled in with satelites' data
	cl_mem bufSat ;
	bufSat = clCreateBuffer(context, CL_MEM_READ_ONLY, SATELITE_COUNT*sizeof(satelite), NULL, &status);
	printf("%d\n",status);
	if (status < 0){
		
		perror("Could not create buffer for satelites");
		//exit(1);
	}
	
	
	
	//Create a buffer that will filled in with pixels ' data
	cl_mem bufPix;
	bufPix = clCreateBuffer(context, CL_MEM_READ_WRITE,SIZE*sizeof(color),NULL, &status);
	printf("%d\n",status);
	if (status < 0){
		perror("Could not create buffer for pixels");
		//exit(1);
	} 
	
	
	// Create a command queue and associate it with the device 
    	
    	cmd_Queue = clCreateCommandQueue(context, devices, 0, &status);
	printf("%d\n",status);
	if (status < 0){
		perror("Could not create a command queue");
		//exit(1);
	} 	
	
	
	/*
	satelite *satelites = NULL;		//input array
	floatvector *pixel = NULL; // input array 
	color *pixels = NULL;		//output array
	
	//#define SIZE WINDOW_HEIGHT*WINDOW_HEIGHT
	// 1024 * 1024
	const int elements = SIZE;
	
	
	//Thess datasize in byte
	size_t datasize_sat = sizeof(satelite)*elements;
	size_t datasize_floatvct = sizeof(floatvector)*elements;
	size_t datasize_col = sizeof(color)*elements;
	// Allocate space for input/output data
	//satelite* satelites = (satelite*)malloc(datasize);
	
	//allocate dynamic memory in C .
	//struct st *x = malloc(sizeof *x)
	// x: must be pointer , no cast require, include appropriate header (stdlib.h)
	//https://www.learn-c.org/en/Dynamic_allocation
	
	
	//ref : alllocate dynamic memory in OpenCL
	//https://stackoverflow.com/questions/14259228/memory-object-allocation-in-opencl-for-dynamic-array-in-structure
	
	satelites = (satelite*)malloc(datasize_sat);
	pixel = (floatvector*)malloc(datasize_floatvct);
	pixels = (color*)malloc(datasize_col);
	
	
	cl_int status;
	
	cl_uint numPlatforms = 0;
	status = clGetPlatformIDs(0,NULL,&numPlatforms);
	
	
	platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
	
	cl_uint numDevices = 0;
	
    	devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
	
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL , numDevices, devices, NULL);
	 
    	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

	
	// Create a buffer object that will hold the input data 
	 
	// Create a buffer object that will hold the output data 
	 cl_mem bufCol;
	 bufCol = clCreateBuffer(context, CL_MEM_WRITE_ONLY,datasize_col,NULL, &status);
	

	// Create a program with source code
    	cl_program program = clCreateProgramWithSource(context, 1, 
        &programSource, NULL, &status);
	  
	
    	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

	
    	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufCol);
	
	// Define an index space (global work size) of work 
    	// items for execution. A workgroup size (local work size) 
    	// is not required, but can be used.
    	size_t globalWorkSize[1];   
 
    	// There are 'elements' work-items 
    	globalWorkSize[0] = elements;
		    

	//Free host resource
	free(satelites);
	free(pixel);
	free(pixels);
	free(platforms);
	free(devices);
	
	
	
	
	//return 0;
  */





// ## You are asked to make this code parallel ##
// Physics engine loop. (This is called once a frame before graphics engine) 
// Moves the satelites based on gravity
// This is done multiple times in a frame because the Euler integration 
// is not accurate enough to be done only once

void parallelPhysicsEngine(){

  size_t global_size = SATELITE_COUNT;

   // Execute the kernel for execution
  status = clEnqueueNDRangeKernel(physics_cmd_queue,physics_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);

  // Wait for Physics Engine loop to finish.
  if (frameNumber < 2){
    clFinish(physics_cmd_queue);
  }
  
}

 
// ## You are asked to make this code parallel ##
// Rendering loop (This is called once a frame after physics engine) 
// Decides the color for each pixel.


void parallelGraphicsEngine(){
  
  status = clWaitForEvents(1, &kernel_events);

  size_t global_size[2] = {WINDOW_HEIGHT, WINDOW_WIDTH};

  //write input array pixel to the device buffer graphics_satelites_buff 
  status = clEnqueueWriteBuffer(graphics_cmd_queue, graphics_satelites_buff, CL_TRUE, 0, TOTAL_SATELLITE_SIZE, satelites, 0, NULL, NULL);
  clFinish(graphics_cmd_queue);

  // Execute the kernel for execution
  status = clEnqueueNDRangeKernel(graphics_cmd_queue, graphics_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
  clFinish(graphics_cmd_queue);


  // Read the device output buffer to the host output array pixels_buff
  status = clEnqueueReadBuffer(graphics_cmd_queue, pixels_buff, CL_TRUE, 0, TOTAL_PIXEL_SIZE, pixels, 0, NULL, NULL);

  clFlush(graphics_cmd_queue);
  clFinish(graphics_cmd_queue);
}

  
  /*
	//write input array satelite to the device buffer bufSat
	status = clEnqueueWriteBuffer(cmd_Queue, bufSat, CL_FALSE, 0, SATELITE_COUNT*sizeof(satelite), satelites, 0, NULL, NULL);
	printf("%d\n",status);
	if (status < 0){
	
		perror("Could not write satelites'data to buffer ");
		//exit(1);
	}
	
	
	
	//write input array pixel to the device buffer bufPix	
	status = clEnqueueWriteBuffer(cmd_Queue, bufPix, CL_FALSE, 0, SIZE*sizeof(color), pixels, 0, NULL, NULL);
	printf("%d\n",status);
	if (status < 0){
	
		perror("Could not write pixels'data to buffer ");
		//exit(1);
	}
	
	
	// Associate the input and output buffers with the kernel 
    	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufSat);
	printf("%d\n",status);
	if (status < 0 ){
	
        	perror("Couldn't set satelite kernel Arg");
        	//exit(1);
   	}

    	
    	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufPix);
	printf("%d\n",status);
	if (status < 0 ){
	
        	perror("Couldn't set pixel kernel Arg");
        	//exit(1);
   	}
   	
   	
   	int global_size_x , global_size_y;
   	global_size_x = ceil((float)WINDOW_WIDTH/(float)WG_SIZE_X)*WG_SIZE_X;
   	global_size_y = ceil((float)WINDOW_WIDTH/(float)WG_SIZE_Y)*WG_SIZE_Y;
   	size_t globalWorkSize[] = {global_size_x, global_size_y};
   	size_t localWorkSize[] = {WG_SIZE_X,WG_SIZE_Y};
   	
	// Execute the kernel for execution
	status = clEnqueueNDRangeKernel(cmd_Queue, kernel, 1, NULL, globalWorkSize,localWorkSize, 0, NULL, NULL);
	printf("%d\n",status);
	if (status < 0){
	
		perror("Could not execute kernel");
		//exit(1);
	}
	
	// Read the device output buffer to the host output array
    	clEnqueueReadBuffer(cmd_Queue, bufPix, CL_TRUE, 0, SIZE* sizeof(color), pixels, 0, NULL, NULL);	
 
	
  */



// ## You may add your own destrcution routines here ##
void destroy(){
	
  //Free OpenCL resource

  clReleaseKernel(physics_kernel);
  clReleaseKernel(graphics_kernel);
  clReleaseCommandQueue(physics_cmd_queue);
  clReleaseCommandQueue(graphics_cmd_queue);
  clReleaseMemObject(physics_satelites_buff);
  clReleaseMemObject(graphics_satelites_buff);
  clReleaseMemObject(pixels_buff);
  clReleaseProgram(physics_program);
  clReleaseProgram(graphics_program);
  clReleaseContext(graphic_context);
  clReleaseContext(physics_context);

  /*clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmd_Queue);
	clReleaseMemObject(bufSat);
	clReleaseMemObject(bufPix);
	clReleaseContext(context);
	*/
}







////////////////////////////////////////////////
// �� TO NOT EDIT ANYTHING AFTER THIS LINE �� //
////////////////////////////////////////////////

// �� DO NOT EDIT THIS FUNCTION ��
// Sequential rendering loop used for finding errors
void sequentialGraphicsEngine(){

    // Graphics pixel loop
    for(int i = 0 ;i < SIZE; ++i) {

      // Row wise ordering
      floatvector pixel = {.x = i % WINDOW_WIDTH, .y = i / WINDOW_WIDTH};

      // This color is used for coloring the pixel
      color renderColor = {.red = 0.f, .green = 0.f, .blue = 0.f};

      // Find closest satelite
      float shortestDistance = INFINITY;

      float weights = 0.f;
      int hitsSatellite = 0;

      // First Graphics satelite loop: Find the closest satellite.
      for(int j = 0; j < SATELITE_COUNT; ++j){
         floatvector difference = {.x = pixel.x - satelites[j].position.x,
                                   .y = pixel.y - satelites[j].position.y};
         float distance = sqrt(difference.x * difference.x + 
                               difference.y * difference.y);

         if(distance < SATELITE_RADIUS) {
            renderColor.red = 1.0f;
            renderColor.green = 1.0f;
            renderColor.blue = 1.0f;
            hitsSatellite = 1;
            break;
         } else {
            float weight = 1.0f / (distance*distance*distance*distance);
            weights += weight;
            if(distance < shortestDistance){
               shortestDistance = distance;
               renderColor = satelites[j].identifier;
            }
         }
      }

      // Second graphics loop: Calculate the color based on distance to every satelite.
      if (!hitsSatellite) {
         for(int j = 0; j < SATELITE_COUNT; ++j){
            floatvector difference = {.x = pixel.x - satelites[j].position.x,
                                      .y = pixel.y - satelites[j].position.y};
            float dist2 = (difference.x * difference.x +
                           difference.y * difference.y);
            float weight = 1.0f/(dist2* dist2);

            renderColor.red += (satelites[j].identifier.red *
                                weight /weights) * 3.0f;

            renderColor.green += (satelites[j].identifier.green *
                                  weight / weights) * 3.0f;

            renderColor.blue += (satelites[j].identifier.blue *
                                 weight / weights) * 3.0f;
         }
      }
      correctPixels[i] = renderColor;
    }
}

void sequentialPhysicsEngine(satelite *s){

   // double precision required for accumulation inside this routine,
   // but float storage is ok outside these loops.
   doublevector tmpPosition[SATELITE_COUNT];
   doublevector tmpVelocity[SATELITE_COUNT];

   for (int i = 0; i < SATELITE_COUNT; ++i) {
       tmpPosition[i].x = s[i].position.x;
       tmpPosition[i].y = s[i].position.y;
       tmpVelocity[i].x = s[i].velocity.x;
       tmpVelocity[i].y = s[i].velocity.y;
   }

   // Physics iteration loop
   for(int physicsUpdateIndex = 0;
       physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
      ++physicsUpdateIndex){

       // Physics satelite loop
      for(int i = 0; i < SATELITE_COUNT; ++i){

         // Distance to the blackhole
         // (bit ugly code because C-struct cannot have member functions)
         doublevector positionToBlackHole = {.x = tmpPosition[i].x -
            HORIZONTAL_CENTER, .y = tmpPosition[i].y - VERTICAL_CENTER};
         double distToBlackHoleSquared =
            positionToBlackHole.x * positionToBlackHole.x +
            positionToBlackHole.y * positionToBlackHole.y;
         double distToBlackHole = sqrt(distToBlackHoleSquared);

         // Gravity force
         doublevector normalizedDirection = {
            .x = positionToBlackHole.x / distToBlackHole,
            .y = positionToBlackHole.y / distToBlackHole};
         double accumulation = GRAVITY / distToBlackHoleSquared;

         // Delta time is used to make velocity same despite different FPS
         // Update velocity based on force
         tmpVelocity[i].x -= accumulation * normalizedDirection.x *
            DELTATIME / PHYSICSUPDATESPERFRAME;
         tmpVelocity[i].y -= accumulation * normalizedDirection.y *
            DELTATIME / PHYSICSUPDATESPERFRAME;

         // Update position based on velocity
         tmpPosition[i].x +=
            tmpVelocity[i].x * DELTATIME / PHYSICSUPDATESPERFRAME;
         tmpPosition[i].y +=
            tmpVelocity[i].y * DELTATIME / PHYSICSUPDATESPERFRAME;
      }
   }

   // double precision required for accumulation inside this routine,
   // but float storage is ok outside these loops.
   // copy back the float storage.
   for (int i = 0; i < SATELITE_COUNT; ++i) {
       s[i].position.x = tmpPosition[i].x;
       s[i].position.y = tmpPosition[i].y;
       s[i].velocity.x = tmpVelocity[i].x;
       s[i].velocity.y = tmpVelocity[i].y;
   }
}

// Just some value that barely passes for OpenCL example program
#define ALLOWED_FP_ERROR 0.08
// �� DO NOT EDIT THIS FUNCTION ��
void errorCheck(){
   for(unsigned int i=0; i < SIZE; ++i) {
      if(fabs(correctPixels[i].red - pixels[i].red) > ALLOWED_FP_ERROR ||
         fabs(correctPixels[i].green - pixels[i].green) > ALLOWED_FP_ERROR ||
         fabs(correctPixels[i].blue - pixels[i].blue) > ALLOWED_FP_ERROR) {
         printf("Buggy pixel at (x=%i, y=%i). Press enter to continue.\n", i % WINDOW_WIDTH, i / WINDOW_WIDTH);
         getchar();
         return;
       }
   }
   printf("Error check passed!\n");
}

// �� DO NOT EDIT THIS FUNCTION ��
void compute(void){
   int timeSinceStart = glutGet(GLUT_ELAPSED_TIME);
   previousFrameTimeSinceStart = timeSinceStart;

   // Error check during first frames
   if (frameNumber < 2) {
      memcpy(backupSatelites, satelites, sizeof(satelite) * SATELITE_COUNT);
      sequentialPhysicsEngine(backupSatelites);
   }
   parallelPhysicsEngine();
   if (frameNumber < 2) {
      for (int i = 0; i < SATELITE_COUNT; i++) {
         if (memcmp (&satelites[i], &backupSatelites[i], sizeof(satelite))) {
            printf("Incorrect satelite data of satelite: %d\n", i);
            getchar();
         }
      }
   }

   int sateliteMovementMoment = glutGet(GLUT_ELAPSED_TIME);
   int sateliteMovementTime = sateliteMovementMoment  - timeSinceStart;

   // Decides the colors for the pixels
   parallelGraphicsEngine();

   int pixelColoringMoment = glutGet(GLUT_ELAPSED_TIME);
   int pixelColoringTime =  pixelColoringMoment - sateliteMovementMoment;

   // Sequential code is used to check possible errors in the parallel version
   if(frameNumber < 2){
      sequentialGraphicsEngine();
      errorCheck();
   }

   int finishTime = glutGet(GLUT_ELAPSED_TIME);
   // Print timings
   int totalTime = finishTime - previousFinishTime;
   previousFinishTime = finishTime;

   printf("Total frametime: %ims, satelite moving: %ims, space coloring: %ims.\n",
      totalTime, sateliteMovementTime, pixelColoringTime);

   // Render the frame
   glutPostRedisplay();
}

// �� DO NOT EDIT THIS FUNCTION ��
// Probably not the best random number generator
float randomNumber(float min, float max){
   return (rand() * (max - min) / RAND_MAX) + min;
}

// DO NOT EDIT THIS FUNCTION
void fixedInit(unsigned int seed){

   if(seed != 0){
     srand(seed);
   }

   // Init pixel buffer which is rendered to the widow
   pixels = (color*)malloc(sizeof(color) * SIZE);

   // Init pixel buffer which is used for error checking
   correctPixels = (color*)malloc(sizeof(color) * SIZE);

   backupSatelites = (satelite*)malloc(sizeof(satelite) * SATELITE_COUNT);


   // Init satelites buffer which are moving in the space
   satelites = (satelite*)malloc(sizeof(satelite) * SATELITE_COUNT);

   // Create random satelites
   for(int i = 0; i < SATELITE_COUNT; ++i){

      // Random reddish color
      color id = {.red = randomNumber(0.f, 0.15f) + 0.1f,
                  .green = randomNumber(0.f, 0.14f) + 0.0f,
                  .blue = randomNumber(0.f, 0.16f) + 0.0f};
    
      // Random position with margins to borders
      floatvector initialPosition = {.x = HORIZONTAL_CENTER - randomNumber(50, 320),
                              .y = VERTICAL_CENTER - randomNumber(50, 320) };
      initialPosition.x = (i / 2 % 2 == 0) ?
         initialPosition.x : WINDOW_WIDTH - initialPosition.x;
      initialPosition.y = (i < SATELITE_COUNT / 2) ?
         initialPosition.y : WINDOW_HEIGHT - initialPosition.y;

      // Randomize velocity tangential to the balck hole
      floatvector positionToBlackHole = {.x = initialPosition.x - HORIZONTAL_CENTER,
                                    .y = initialPosition.y - VERTICAL_CENTER};
      float distance = (0.06 + randomNumber(-0.01f, 0.01f))/ 
        sqrt(positionToBlackHole.x * positionToBlackHole.x + 
          positionToBlackHole.y * positionToBlackHole.y);
      floatvector initialVelocity = {.x = distance * -positionToBlackHole.y,
                                .y = distance * positionToBlackHole.x};

      // Every other orbits clockwise
      if(i % 2 == 0){
         initialVelocity.x = -initialVelocity.x;
         initialVelocity.y = -initialVelocity.y;
      }

      satelite tmpSatelite = {.identifier = id, .position = initialPosition,
                              .velocity = initialVelocity};
      satelites[i] = tmpSatelite;
   }
}

// �� DO NOT EDIT THIS FUNCTION ��
void fixedDestroy(void){
   destroy();

   free(pixels);
   free(correctPixels);
   free(satelites);

   if(seed != 0){
     printf("Used seed: %i\n", seed);
   }
}

// �� DO NOT EDIT THIS FUNCTION ��
// Renders pixels-buffer to the window 
void render(void){
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_FLOAT, pixels);
   glutSwapBuffers();
   frameNumber++;
}

// DO NOT EDIT THIS FUNCTION
// Inits glut and start mainloop
int main(int argc, char** argv){

   if(argc > 1){
     seed = atoi(argv[1]);
     printf("Using seed: %i\n", seed);
   }

   // Init glut window
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
   glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
   glutCreateWindow("Parallelization excercise");
   glutDisplayFunc(render);
   atexit(fixedDestroy);
   previousFrameTimeSinceStart = glutGet(GLUT_ELAPSED_TIME);
   previousFinishTime = glutGet(GLUT_ELAPSED_TIME);
   glEnable(GL_DEPTH_TEST);
   glClearColor(0.0, 0.0, 0.0, 1.0);
   fixedInit(seed);
   init();

   // compute-function is called when everythin from last frame is ready
   glutIdleFunc(compute);

   // Start main loop
   glutMainLoop();
}
