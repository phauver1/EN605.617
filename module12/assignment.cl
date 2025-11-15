__kernel void bufferSaxpy(__global uchar* buffer, const unsigned int numElems)
{
	size_t id = get_global_id(0);
    __global float* a = (__global float*)buffer;
    __global float4* x = (__global float4*)(a + numElems);
    __global float4* y = (__global float4*)(x + numElems);

    y[id] = (float4)a[id] * x[id] + y[id];
}