
#pragma once
#ifndef CUTIL_MATH_H
#define CUTIL_MATH_H

#include <cuda_runtime.h>
inline __host__ __device__ float3 make_float3(float s)
{
	return make_float3(s, s, s);
}

inline __host__ __device__ float3 normalize(float3 v)
{
	//float invLen = rsqrtf(dot(v, v));
	return v; //* invLen;
}
////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline __host__ __device__ float3 operator*(float3 a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

#endif