#define GRID_SIZE (1LL << 24)
#define BLOCK_SIZE 512
#define CHUNK_SIZE (GRID_SIZE / BLOCK_SIZE)
#define RNG_MUL 25214903917ULL
#define RNG_ADD 11ULL
#define RNG_MASK ((1ULL << 48) - 1)

#ifndef BEGIN
#define BEGIN 0
#endif
#ifndef END
#define END (1ULL << 48)
#endif
#ifndef CACTUS_HEIGHT
#define CACTUS_HEIGHT 9
#endif
#ifndef FLOOR_LEVEL
#define FLOOR_LEVEL 62
#endif

#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>

#include <cuda.h>

#ifdef BOINC
  #include "boinc_api.h"
#if defined _WIN32 || defined _WIN64
  #include "boinc_win.h"
#endif
#endif

__device__ uint64_t block_add_gpu[BLOCK_SIZE + 1];
__device__ uint64_t block_mul_gpu[BLOCK_SIZE + 1];
__device__ uint64_t chunk_add_gpu[CHUNK_SIZE + 1];
__device__ uint64_t chunk_mul_gpu[CHUNK_SIZE + 1];

__device__ inline int32_t next(uint32_t *random, uint32_t *index, int bits)
{
	return (random[(*index)++] >> (32 - bits));
}

__device__ inline int32_t next_int(uint32_t *random, uint32_t *index, int32_t bound)
{
	int32_t bits, value;
	do {
		bits = next(random, index, 31);
		value = bits % bound;
	} while (bits - value + (bound - 1) < 0);
	return value;
}

__device__ inline int32_t next_int_unknown(uint32_t *random, uint32_t *index, int32_t bound)
{
	if ((bound & -bound) == bound) {
		return (int32_t) ((bound * (uint64_t) next(random, index, 31)) >> 31);
	} else {
		return next_int(random, index, bound);
	}
}

__device__ inline uint8_t extract(const uint32_t *heightmap, uint16_t pos)
{
	return ((heightmap[pos >> 3] >> ((pos & 7) << 2)) & 15) + FLOOR_LEVEL;
}

__device__ inline void increase(uint32_t *heightmap, uint16_t pos, uint8_t addend)
{
	heightmap[pos >> 3] += addend << ((pos & 7) << 2);
}

__global__ void crack(uint64_t seed, uint64_t *out, uint64_t *out_n)
{
	__shared__ uint32_t random[BLOCK_SIZE + 1024];
	__shared__ uint32_t skip_index[BLOCK_SIZE + 1024 - 100];
	__shared__ uint32_t skip_first[BLOCK_SIZE + 1024 - 102];
	__shared__ uint32_t skip_always[BLOCK_SIZE + 1024 - 102];
	__shared__ uint32_t floor_skip[BLOCK_SIZE + 1024 - 102];
	__shared__ uint8_t floor_terrain[BLOCK_SIZE + 1024 - 102];
	__shared__ uint32_t offset_skip[BLOCK_SIZE + 1024 - 4];
	__shared__ uint8_t offset_height[BLOCK_SIZE + 1024 - 4];
	uint32_t heightmap[128];
	uint32_t random_index;

	seed = (seed * chunk_mul_gpu[blockIdx.x] + chunk_add_gpu[blockIdx.x]) & RNG_MASK;
	seed = (seed * block_mul_gpu[threadIdx.x] + block_add_gpu[threadIdx.x]) & RNG_MASK;
	uint64_t seed2 = seed;
	seed = ((seed - 11ULL) * 246154705703781ULL) & RNG_MASK;
	random[threadIdx.x + BLOCK_SIZE * 0] = (uint32_t) (seed2 >> 16);
	for (int i = threadIdx.x + BLOCK_SIZE; i < BLOCK_SIZE + 1024; i += BLOCK_SIZE) {
		seed2 = (seed2 * block_mul_gpu[BLOCK_SIZE] + block_add_gpu[BLOCK_SIZE]) & RNG_MASK;
		random[i] = (uint32_t) (seed2 >> 16);
	}
	for (int i = 0; i < 128; i++) {
		heightmap[i] = 0;
	}
	__syncthreads();

	for (int i = threadIdx.x; i < BLOCK_SIZE + 1024 - 4; i += BLOCK_SIZE) {
		random_index = i;
		uint8_t offset = next_int_unknown(random, &random_index, next_int(random, &random_index, 3) + 1) + 1;
		offset_height[i] = offset;
		offset_skip[i] = random_index;
	}
	__syncthreads();

	for (int i = threadIdx.x; i < BLOCK_SIZE + 1024 - 100; i += BLOCK_SIZE) {
		random_index = i;
		for (int j = 0; j < 10; j++) {
			random_index += 6;
			random_index = offset_skip[random_index];
		}
		skip_index[i] = random_index;
	}
	__syncthreads();

	for (int i = threadIdx.x; i < BLOCK_SIZE + 1024 - 102; i += BLOCK_SIZE) {
		random_index = i + 2;
		int16_t terrain = next_int_unknown(random, &random_index, (FLOOR_LEVEL + 1) * 2);
		floor_terrain[i] = terrain;
		floor_skip[i] = random_index;
		if (terrain - 3 > FLOOR_LEVEL + CACTUS_HEIGHT + 1) {
			skip_first[i] = skip_index[random_index];
			skip_always[i] = skip_index[random_index];
		} else if (terrain - 3 > FLOOR_LEVEL + 1) {
			skip_first[i] = skip_index[random_index];
			skip_always[i] = 0;
		} else if (terrain + 3 <= FLOOR_LEVEL && terrain - 3 >= 0) {
			skip_first[i] = random_index + 60;
			skip_always[i] = random_index + 60;
		} else {
			skip_first[i] = 0;
			skip_always[i] = 0;
		}
	}
	__syncthreads();

	random_index = threadIdx.x;
	uint16_t best = 0;
	bool changed = false;
	int i = 0;
	for (; i < 10 && skip_first[random_index]; i++) {
		random_index = skip_first[random_index];
	}
	for (; i < 10; i++) {
		if (!changed && skip_first[random_index]) {
			random_index = skip_first[random_index];
			continue;
		}
		uint16_t bx = next(random, &random_index, 4) + 8;
		uint16_t bz = next(random, &random_index, 4) + 8;
		uint16_t initial = bx * 32 + bz;
		int16_t terrain;
		if (extract(heightmap, initial) == FLOOR_LEVEL) {
			if (skip_always[random_index - 2]) {
				random_index = skip_always[random_index - 2];
				continue;
			}
			terrain = floor_terrain[random_index - 2];
			random_index = floor_skip[random_index - 2];
		} else {
			terrain = next_int_unknown(random, &random_index, (extract(heightmap, initial) + 1) * 2);
			if (terrain + 3 <= FLOOR_LEVEL && terrain - 3 >= 0) {
				random_index += 60;
				continue;
			}
		}
		if (terrain - 3 > extract(heightmap, best) + 1) {
			random_index = skip_index[random_index];
			continue;
		}
		for (int j = 0; j < 10; j++) {
			int16_t bx = next(random, &random_index, 3) - next(random, &random_index, 3);
			int16_t by = next(random, &random_index, 2) - next(random, &random_index, 2);
			int16_t bz = next(random, &random_index, 3) - next(random, &random_index, 3);
			uint16_t xz = initial + bx * 32 + bz;
			int16_t y = (int16_t) terrain + by;
			if (y <= extract(heightmap, xz) && y >= 0) continue;
			uint8_t offset = offset_height[random_index];
			random_index = offset_skip[random_index];
			if (y != extract(heightmap, xz) + 1) continue;
			if (y == FLOOR_LEVEL + 1) {
				uint8_t mask = 0;
				if (bz != 0x00) mask |= extract(heightmap, xz - 1) - FLOOR_LEVEL;
				if (bz != 0x1F) mask |= extract(heightmap, xz + 1) - FLOOR_LEVEL;
				if (bx != 0x00) mask |= extract(heightmap, xz - 32) - FLOOR_LEVEL;
				if (bx != 0x1F) mask |= extract(heightmap, xz + 32) - FLOOR_LEVEL;
				if (mask) continue;
			}
			increase(heightmap, xz, offset);
			changed = true;
			if (extract(heightmap, xz) > extract(heightmap, best)) best = xz;
		}
	}
	if (extract(heightmap, best) - FLOOR_LEVEL >= CACTUS_HEIGHT) {
		out[atomicAdd((unsigned long long*) out_n, 1ULL)] = seed;
	}
}

uint64_t block_add[BLOCK_SIZE + 1];
uint64_t block_mul[BLOCK_SIZE + 1];
uint64_t chunk_add[CHUNK_SIZE + 1];
uint64_t chunk_mul[CHUNK_SIZE + 1];
uint64_t offset = 0;
uint64_t seed = 0;
uint64_t total_seeds = 0;
std::mutex mutexcuda;
std::thread threads[1];

void run(int gpu_device)
{
	uint64_t *out;
	uint64_t *out_n;
	cudaSetDevice(gpu_device);
	cudaMallocManaged(&out, GRID_SIZE * sizeof(*out));
	cudaMallocManaged(&out_n, sizeof(*out_n));
	cudaMemcpyToSymbol(block_add_gpu, block_add, (BLOCK_SIZE + 1) * sizeof(*block_add));
	cudaMemcpyToSymbol(block_mul_gpu, block_mul, (BLOCK_SIZE + 1) * sizeof(*block_mul));
	cudaMemcpyToSymbol(chunk_add_gpu, chunk_add, (CHUNK_SIZE + 1) * sizeof(*chunk_add));
	cudaMemcpyToSymbol(chunk_mul_gpu, chunk_mul, (CHUNK_SIZE + 1) * sizeof(*chunk_mul));

	while (true) {
		*out_n = 0;
		{
			std::lock_guard<std::mutex> lock(mutexcuda);
			if (offset >= END) break;
			uint64_t seed_gpu = (seed * RNG_MUL + RNG_ADD) & RNG_MASK;
			crack<<<CHUNK_SIZE, BLOCK_SIZE>>>(seed_gpu, out, out_n);
			offset += GRID_SIZE;
			seed = (seed * chunk_mul[CHUNK_SIZE] + chunk_add[CHUNK_SIZE]) & RNG_MASK;
		}
		cudaDeviceSynchronize();
		{
			std::lock_guard<std::mutex> lock(mutexcuda);
			total_seeds += *out_n;
			for (uint64_t i = 0; i < *out_n; i++)
				fprintf(stderr,"%lu\n", out[i]);
			fflush(stderr);
		}
	}

	cudaFree(out_n);
	cudaFree(out);
}

int main()
{
	#ifdef BOINC
	BOINC_OPTIONS options;

	boinc_options_defaults(options);
	options.normal_thread_priority = true;
	boinc_init_options(&options);
	#endif
	
	block_add[0] = 0;
	block_mul[0] = 1;
	for (uint64_t i = 0; i < BLOCK_SIZE; i++) {
		block_add[i + 1] = (block_add[i] * RNG_MUL + RNG_ADD) & RNG_MASK;
		block_mul[i + 1] = (block_mul[i] * RNG_MUL) & RNG_MASK;
	}

	chunk_add[0] = 0;
	chunk_mul[0] = 1;
	for (uint64_t i = 0; i < CHUNK_SIZE; i++) {
		chunk_add[i + 1] = (chunk_add[i] * block_mul[BLOCK_SIZE] + block_add[BLOCK_SIZE]) & RNG_MASK;
		chunk_mul[i + 1] = (chunk_mul[i] * block_mul[BLOCK_SIZE]) & RNG_MASK;
	}

	for (; offset + GRID_SIZE <= BEGIN; offset += GRID_SIZE)
		seed = (seed * chunk_mul[CHUNK_SIZE] + chunk_add[CHUNK_SIZE]) & RNG_MASK;
	for (; offset + 1 <= BEGIN; offset += 1)
		seed = (seed * RNG_MUL + RNG_ADD) & RNG_MASK;
	
	int gpu_device = 0;
	
	#ifdef BOINC
	APP_INIT_DATA aid;
	boinc_get_init_data(aid);
	
	if (aid.gpu_device_num >= 0) {
		gpu_device = aid.gpu_device_num;
		fprintf(stderr,"boinc gpu %i gpuindex: %i \n", aid.gpu_device_num, gpu_device);
		} else {
		fprintf(stderr,"stndalone gpuindex %i \n", gpu_device);
	}
	#endif

	threads[0] = std::thread(run, gpu_device);

	time_t start_time = time(NULL);
	while (offset < END) {
		using namespace std::chrono_literals;
		std::this_thread::sleep_for(1s);
		time_t elapsed = time(NULL) - start_time;
		uint64_t count = offset - BEGIN;
		double frac = (double) count / (double) (END - BEGIN);
		double done = (double) count / 1000000.0;
		double speed = done / (double) elapsed;
		boinc_fraction_done(frac);
		fprintf(stderr, "%10.2fb %7lis %8.2fm/s %6.2f%% =%-6lu\n", done / 1000, elapsed, speed, frac * 100.0, total_seeds);
	}
	for (std::thread& thread : threads)
		thread.join();
		
	boinc_finish(0);
}
