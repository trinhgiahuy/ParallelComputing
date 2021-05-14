// These are used to decide the window size
#define WINDOW_HEIGHT 1024
#define WINDOW_WIDTH 1024

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

// Stores rendered colors. Each float may vary from 0.0f ... 1.0f
typedef struct{
	float red;
	float green;
	float blue;
} color;

// Stores 2D data like the coordinates
typedef struct{
   double x;
   double y;
} doublevector;


// Stores the satelite data, which fly around black hole in the space
typedef struct{
	color identifier;
	floatvector position;
	floatvector velocity;
} satelite;


__kernel void parallelPhysicsEngineKernel(__global satelite *satelites)
{
	//Get global index as globa_id
	size_t id_global = get_global_id(0);
	
	__private doublevector tmpVelocity;
	__private doublevector tmpPosition;

	tmpPosition.x = satelites[id_global].position.x;
	tmpPosition.y = satelites[id_global].position.y;
	tmpVelocity.x = satelites[id_global].velocity.x;
	tmpVelocity.y = satelites[id_global].velocity.y;
	
	// Physics iteration loop
   	for(int physicsUpdateIndex = 0; physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
	++physicsUpdateIndex){
		
		// Distance to the blackhole (bit ugly code because C-struct cannot have member
		//functions)
		
		doublevector positionToBlackHole = {.x = tmpPosition.x-HORIZONTAL_CENTER,
		.y = tmpPosition.y - VERTICAL_CENTER};
		double distToBlackHoleSquared =
		positionToBlackHole.x * positionToBlackHole.x +
		positionToBlackHole.y* positionToBlackHole.y;

		double distToBlackHole = sqrt(distToBlackHoleSquared);

		// Gravity force
		doublevector normalizedDirection = {
		.x = positionToBlackHole.x / distToBlackHole,
		.y = positionToBlackHole.y / distToBlackHole};
		double accumulation = GRAVITY / distToBlackHoleSquared;

		// Delta time is used to make velocity same despite different FPS
		// Update velocity based on force
		tmpVelocity.x -= accumulation * normalizedDirection.x *
		DELTATIME / PHYSICSUPDATESPERFRAME;
		tmpVelocity.y -= accumulation * normalizedDirection.y *
		DELTATIME / PHYSICSUPDATESPERFRAME;


         	// Update position based on velocity
         	tmpPosition.x +=
            	tmpVelocity.x * DELTATIME / PHYSICSUPDATESPERFRAME;
         	tmpPosition.y +=
            	tmpVelocity.y * DELTATIME / PHYSICSUPDATESPERFRAME;
	}

	// double precision required for accumulation inside this routine,
    	// but float storage is ok outside these loops.
    	// copy back the float storage.
	satelites[id_global].position.x = tmpPosition.x;
       	satelites[id_global].position.y = tmpPosition.y;
       	satelites[id_global].velocity.x = tmpVelocity.x;
	satelites[id_global].velocity.y = tmpVelocity.y;
}


__kernel void parallelGraphicsEngineKernel(__global satelite *satelites, __global color* pixels){


	 size_t id_x = get_global_id(1);
	 size_t id_y = get_global_id(0);
	 

	// Row wise ordering
	__private floatvector pixel = {.x = id_x, .y = id_y};

	//This color is used for coloring the pixel
	__private color renderColor = {.red = 0.f, .green = 0.f, .blue = 0.f};
	__private color incrementColor = { .red = 0.f, .green = 0.f, .blue = 0.f};


	// Find closest satelite
	float shortestDistance = INFINITY;

	float weights = 0.f;
	int hitsSatellite = 0;



	// First Graphics satelite loop: Find the closest satellite.
	for(int j = 0; j < SATELITE_COUNT; ++j) {
		floatvector difference = {.x = pixel.x - satelites[j].position.x,
					.y = pixel.y - satelites[j].position.y};
		float distance = sqrt(difference.x * difference.x + 
							difference.y * difference.y);

		float weight = 1.0f / (distance*distance*distance*distance);
		weights += weight;
	
		
		if (distance < shortestDistance){
		   shortestDistance = distance;
		   renderColor = satelites[j].identifier;
		}

		incrementColor.red += satelites[j].identifier.red * weight;
		incrementColor.green += satelites[j].identifier.green * weight;
        	incrementColor.blue += satelites[j].identifier.blue * weight;
        
    	}
						
	
   	// Calculate the color based on distance to every satelite.
	if(shortestDistance < SATELITE_RADIUS) {

		    renderColor.red = 1.0f;
		    renderColor.green = 1.0f;
		    renderColor.blue = 1.0f;

	}else {
	            
        	    renderColor.red   += incrementColor.red / weights * 3.0f;                                                      
        	    renderColor.green += incrementColor.green / weights * 3.0f;             	 
	            renderColor.blue  += incrementColor.blue / weights * 3.0f;
	}



	pixels[id_x + WINDOW_WIDTH * id_y] = renderColor;

}
