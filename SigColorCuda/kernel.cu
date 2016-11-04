#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include "filter.h"
#include <string>
#include <sstream>
#include <fstream>

template <typename T>
__device__ void cuYC48_RGB48(T* px)
{
	float y = static_cast<float>(px[0]);
	float cb = static_cast<float>(px[1]);
	float cr = static_cast<float>(px[2]);
	float r = 1.0037736040867458f*y + 0.0009812686948862392f*cb + 1.4028706125758748f*cr;
	float g = 1.0031713814217937f*y + -0.34182057237626395f*cb + -0.7126004638855613f*cr;
	float b = 1.0038646965904563f*y + 1.7738420513779833f*cb + 0.0018494308641594699f*cr;
	px[0] = static_cast<T>(r);
	px[1] = static_cast<T>(g);
	px[2] = static_cast<T>(b);
}

template <typename T>
__device__ void cuRGB48_YC48(T* px)
{
	float r = static_cast<float>(px[0]);
	float g = static_cast<float>(px[1]);
	float b = static_cast<float>(px[2]);
	float y = 0.297607421875f*r + 0.586181640625f*g + 0.11279296875f*b;
	float cb = -0.1689453125f*r + -0.331298828125f*g + 0.5f*b;
	float cr = 0.5f*r + -0.419189453125f*g + -0.0810546875f*b;
	px[0] = static_cast<T>(y);
	px[1] = static_cast<T>(cb);
	px[2] = static_cast<T>(cr);
}

template <typename T>
__device__ void cuSig(T* px, float midtone, float strength)
{
	float i = static_cast<float>(*px) / 4096.0f;
	float a = midtone;
	float b = strength;
	float term_a, term_b, term_c;
	term_a = 1.0f / (1.0f + expf(b*(a - i)));
	term_b = 1.0f / (1.0f + expf(b*(a - 1.0f)));
	term_c = 1.0f / (1.0f + expf(a*b));
	float s = ((term_a - term_c) / (term_b - term_c))*4096.0;
	*px = static_cast<T>(s);
}

template <typename T>
__device__ void cuLogit(T* px, float midtone, float strength)
{
	float i = static_cast<float>(*px) / 4096.0f;
	float a = midtone;
	float b = strength;
	float term_b, term_c;
	term_b = 1.0f / (1.0f + expf(b*(a - 1.0f)));
	term_c = 1.0f / (1.0f + expf(a*b));
	float l = a - (logf(1.0f / (i*(term_b - term_c) + term_c) - 1.0f) / b);

	*px = static_cast<T>(l*4096.0f);
}

__global__ void SigmodialTransform(short* image, size_t width, size_t height, size_t stride, float midtone, float strength, bool r = true, bool g = true, bool b = true)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y* blockDim.y + threadIdx.y;
	if ((x > width) || (y > height)) return;
	short* px = reinterpret_cast<short*>(reinterpret_cast<unsigned char*>(image) + stride*y + x * sizeof(short) * 3);
	float fpx[3] = { static_cast<float>(px[0]), static_cast<float>(px[1]), static_cast<float>(px[2]) };
	cuYC48_RGB48(fpx);
	if (r) cuSig(fpx, midtone, strength);
	if (g) cuSig(fpx+1, midtone, strength);
	if (b) cuSig(fpx+2, midtone, strength);
	cuRGB48_YC48(fpx);
	px[0] = static_cast<short>(fpx[0]);
	px[1] = static_cast<short>(fpx[1]);
	px[2] = static_cast<short>(fpx[2]);
}

__global__ void LogitTransform(short* image, size_t width, size_t height, size_t stride, float midtone, float strength, bool r = true, bool g = true, bool b = true)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y* blockDim.y + threadIdx.y;
	if ((x > width) || (y > height)) return;
	short* px = reinterpret_cast<short*>(reinterpret_cast<unsigned char*>(image) + stride*y + x * sizeof(short) * 3);
	float fpx[3] = { static_cast<float>(px[0]), static_cast<float>(px[1]), static_cast<float>(px[2]) };
	cuYC48_RGB48(fpx);
	if (r) cuLogit(fpx, midtone, strength);
	if (g) cuLogit(fpx+1, midtone, strength);
	if (b) cuLogit(fpx+2, midtone, strength);
	cuRGB48_YC48(fpx);
	px[0] = static_cast<short>(fpx[0]);
	px[1] = static_cast<short>(fpx[1]);
	px[2] = static_cast<short>(fpx[2]);
}


cudaError_t utlSLTransform(void* ycp_edit, int w, int h, int max_w, void* dev_buffer, size_t dev_stride, float midtone, float strength, bool mode = true, bool r=true, bool g=true, bool b=true)
{
	cudaError_t err = cudaMemcpy2D(dev_buffer, dev_stride, ycp_edit, max_w * sizeof(short) * 3, w * sizeof(short) * 3, h, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) return err;
	dim3 threadsperblock(32, 32);
	dim3 numBlocks((w / threadsperblock.x) + 1, (h / threadsperblock.y) + 1);

	if (mode)
	{
		SigmodialTransform <<<numBlocks, threadsperblock>>> (reinterpret_cast<short*>(dev_buffer), w, h, dev_stride, midtone, strength, r, g, b);
	}
	else
	{
		LogitTransform <<<numBlocks, threadsperblock >>> (reinterpret_cast<short*>(dev_buffer), w, h, dev_stride, midtone, strength, r, g, b);
	}
	err = cudaGetLastError();
	if (err != cudaSuccess) return err;
	err = cudaMemcpy2D(ycp_edit, max_w * 3 * sizeof(short), dev_buffer, dev_stride, w * 3 * sizeof(short), h, cudaMemcpyDeviceToHost);
	return err;
}

#ifdef _DEBUG
#define PLUGIN_NAME_S "SContrast CUDA (DEBUG)"
#define PLUGIN_NAME_D "SDeContrast CUDA (DEBUG)"
#define VERSION_S "SContrast CUDA (DEBUG) v0.01 by MT"
#define VERSION_D "SDeContrast CUDA (DEBUG) v0.01 by MT"
#else
#define PLUGIN_NAME_S "SContrast CUDA"
#define PLUGIN_NAME_D "SDeContrast CUDA"
#define VERSION_S "SContrast CUDA v0.01 by MT"
#define VERSION_D "SDeContrast CUDA v0.01 by MT"
#endif

std::ofstream logfilec("sigcontrast_log.csv", std::ios_base::app);
std::ofstream logfiled("sigdecontrast_log.csv", std::ios_base::app);
std::ostringstream logcbuf;
std::ostringstream logdbuf;
char* slider_label[] = { "Midtone", "Strength" };
int slider_default[] = { 50, 5 };
int slider_min[] = { 1, 1 };
int slider_max[] = { 100, 20 };
int trackN = 2;

char* box_label[] = { "R", "G", "B", "Show Benchmark", "Log Benchmark when saving" };
int box_default[] = { 1,1,1,0,0 };
int checkN = 5;

cudaError_t errors = cudaSuccess;
cudaError_t errord = cudaSuccess;
cudaEvent_t start, stop, startd, stopd;
float msc, msd;
void* dev_SC = nullptr;
void* dev_DC = nullptr;
size_t strides = 0;
size_t strided = 0;

FILTER_DLL SC = {
	FILTER_FLAG_EX_INFORMATION | FILTER_FLAG_PRIORITY_LOWEST,	//	filter flags, use bitwise OR to add more
	0, 0,						//	dialogbox size
	PLUGIN_NAME_S,			//	Filter plugin name
	trackN,					//	トラックバーの数 (0なら名前初期値等もNULLでよい)
	slider_label ,					//	slider label names in English
	slider_default,				//	トラックバーの初期値郡へのポインタ
	slider_min, slider_max,			//	トラックバーの数値の下限上限 (NULLなら全て0～256)
	checkN,					//	チェックボックスの数 (0なら名前初期値等もNULLでよい)
	box_label,					//	チェックボックスの名前郡へのポインタ
	box_default,				//	チェックボックスの初期値郡へのポインタ
	func_proc_s,					//	main filter function, use NULL to skip
	func_init_s,						//	initialization function, use NULL to skip
	func_exit_s,						//	on-exit function, use NULL to skip
	func_update_s,						//	invokes when when settings changed. use NULL to skip
	func_WndProc_s,						//	for capturing dialog's control messages. Essential if you use button or auto uncheck some checkboxes.
	NULL, NULL,					//	Reserved. Do not use.
	NULL,						//  pointer to extra data when FILTER_FLAG_EX_DATA is set
	NULL,						//  extra data size
	VERSION_S,
	//  pointer or c-string for full filter info when FILTER_FLAG_EX_INFORMATION is set.
	NULL,						//	invoke just before saving starts. NULL to skip
	func_save_end_s,						//	invoke just after saving ends. NULL to skip
};

FILTER_DLL SD = {
	FILTER_FLAG_EX_INFORMATION | FILTER_FLAG_PRIORITY_LOWEST,	//	filter flags, use bitwise OR to add more
	0, 0,						//	dialogbox size
	PLUGIN_NAME_D,			//	Filter plugin name
	trackN,					//	トラックバーの数 (0なら名前初期値等もNULLでよい)
	slider_label ,					//	slider label names in English
	slider_default,				//	トラックバーの初期値郡へのポインタ
	slider_min, slider_max,			//	トラックバーの数値の下限上限 (NULLなら全て0～256)
	checkN,					//	チェックボックスの数 (0なら名前初期値等もNULLでよい)
	box_label,					//	チェックボックスの名前郡へのポインタ
	box_default,				//	チェックボックスの初期値郡へのポインタ
	func_proc_d,					//	main filter function, use NULL to skip
	func_init_d,						//	initialization function, use NULL to skip
	func_exit_d,						//	on-exit function, use NULL to skip
	func_update_d,						//	invokes when when settings changed. use NULL to skip
	func_WndProc_d,						//	for capturing dialog's control messages. Essential if you use button or auto uncheck some checkboxes.
	NULL, NULL,					//	Reserved. Do not use.
	NULL,						//  pointer to extra data when FILTER_FLAG_EX_DATA is set
	NULL,						//  extra data size
	VERSION_D,
	//  pointer or c-string for full filter info when FILTER_FLAG_EX_INFORMATION is set.
	NULL,						//	invoke just before saving starts. NULL to skip
	func_save_end_d,						//	invoke just after saving ends. NULL to skip
};


FILTER_DLL* pluginlist[] = { &SC, &SD, nullptr };
// Export the above filter table
EXTERN_C  __declspec(dllexport) FILTER_DLL** GetFilterTableList(void)
{

	return pluginlist;
}

BOOL func_init_s(FILTER *fp)
{
	errors = cudaSetDevice(0);
	if (errors != cudaSuccess)
	{
		MessageBox(fp->hwnd, "Cannot set default CUDA device!", "SContrast CUDA Error", MB_OK | MB_ICONERROR);
		return FALSE;
	}
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	return TRUE;
}

BOOL func_proc_s(FILTER *fp, FILTER_PROC_INFO *fpip)
{
	cudaEventRecord(start);
	if (!dev_SC)
	{
		errors = cudaMallocPitch(&dev_SC, &strides, fpip->max_w*3*sizeof(short), fpip->max_h);
		if (errors != cudaSuccess)
		{
			MessageBox(fp->hwnd, "Not enough Video Memory!", "SContrast CUDA Error", MB_OK | MB_ICONERROR);
			return FALSE;
		}
	}
	float midtone = static_cast<float>(fp->track[0]) / 100.0f;
	float strength = static_cast<float>(fp->track[1]);
	errors = utlSLTransform((void*)fpip->ycp_edit, fpip->w, fpip->h, fpip->max_w, dev_SC, strides, midtone, strength, true, (fp->check[0]==1), (fp->check[1]==1), (fp->check[2]==1));
	if (errors != cudaSuccess)
	{
		MessageBox(fp->hwnd, "Kernel Error!", "SContrast CUDA Error", MB_OK | MB_ICONERROR);
		MessageBox(fp->hwnd, cudaGetErrorString(errors), "SContrast CUDA_Error", MB_OK);
		return FALSE;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msc, start, stop);
	
	if (fp->check[3] && !(fp->exfunc->is_saving(fpip->editp)))
	{
		std::string msg = "SCon: " + std::to_string(msc) + "ms @" + std::to_string(fpip->w) + "x" + std::to_string(fpip->h);
		SetWindowText(fp->hwnd, msg.c_str());
	}

	if (fp->check[4] && fp->exfunc->is_saving(fpip->editp))
	{
		// write into buffer
		logcbuf << std::to_string(fpip->frame) << ", " << std::to_string(fpip->w) << ", " << std::to_string(fpip->h) << ", " << std::to_string(msc) << std::endl;
		if (logcbuf.tellp() >= 65536)
		{
			logfilec << logcbuf.str();
			logfilec.flush();
			logcbuf.str(std::string());
		}
	}
	
	return TRUE;
}

BOOL func_exit_s(FILTER *fp)
{
	logfilec.flush();
	logcbuf.flush();
	logfilec.close();
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	//if (dev_SC) errors = cudaFree(dev_SC);
	//if (errors != cudaSuccess)
	//{
	//	MessageBox(fp->hwnd, "cudaFree error on Exit!", "SContrast CUDA Error", MB_OK | MB_ICONERROR);
	//	return FALSE;
	//}
	errors = cudaDeviceReset();
	if (errors != cudaSuccess)
	{
		MessageBox(fp->hwnd, "Device cleanup error!", "SContrast CUDA Error", MB_OK | MB_ICONERROR);
		return FALSE;
	}
	return TRUE;
}

BOOL func_save_end_s(FILTER *fp, void *editp)
{
	if (fp->check[4])
	{
		logfilec << logcbuf.str();
		logfilec.flush();
		logcbuf.str(std::string());
	}
	return TRUE;
}

BOOL func_update_s(FILTER *fp, int status)
{
	switch (status)
	{
		case FILTER_UPDATE_STATUS_CHECK + 3:
		{
			if (fp->check[3] == 0)
			{
				SetWindowText(fp->hwnd, PLUGIN_NAME_S);
				return FALSE;
			}
		}break;

	}
	return TRUE;
}

BOOL func_WndProc_s(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam, void *editp, FILTER *fp)
{
	switch (message)
	{
	case WM_FILTER_FILE_CLOSE:
	{
		if (dev_SC)
		{
			cudaFree(dev_SC);
			dev_SC = nullptr;
		};
		break;
	}
	};
	return FALSE;
}

/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/***************** DeContrast Plugin ****************************************************/
/****************************************************************************************/

BOOL func_init_d(FILTER *fp)
{
	errord = cudaSetDevice(0);
	if (errord != cudaSuccess)
	{
		MessageBox(fp->hwnd, "Cannot set default CUDA device!", "SDeContrast CUDA Error", MB_OK | MB_ICONERROR);
		return FALSE;
	}
	cudaEventCreate(&startd);
	cudaEventCreate(&stopd);
	return TRUE;
}

BOOL func_proc_d(FILTER *fp, FILTER_PROC_INFO *fpip)
{
	cudaEventRecord(startd);
	if (!dev_DC)
	{
		errord = cudaMallocPitch(&dev_DC, &strided, fpip->max_w * 3 * sizeof(short), fpip->max_h);
		if (errord != cudaSuccess)
		{
			MessageBox(fp->hwnd, "Not enough Video Memory!", "SDeContrast CUDA Error", MB_OK | MB_ICONERROR);
			return FALSE;
		}
	}
	float midtone = static_cast<float>(fp->track[0]) / 100.0f;
	float strength = static_cast<float>(fp->track[1]);
	errord = utlSLTransform((void*)fpip->ycp_edit, fpip->w, fpip->h, fpip->max_w, dev_DC, strided, midtone, strength, false, (fp->check[0] == 1), (fp->check[1] == 1), (fp->check[2] == 1));
	if (errord != cudaSuccess)
	{
		MessageBox(fp->hwnd, "Kernel Error!", "SDeContrast CUDA Error", MB_OK | MB_ICONERROR);
		MessageBox(fp->hwnd, cudaGetErrorString(errord), "SDeContrast CUDA_Error", MB_OK);
		return FALSE;
	}
	cudaEventRecord(stopd);
	cudaEventSynchronize(stopd);
	cudaEventElapsedTime(&msd, startd, stopd);

	if (fp->check[3] && !(fp->exfunc->is_saving(fpip->editp)))
	{
		std::string msg = "SDeCon: " + std::to_string(msd) + "ms @" + std::to_string(fpip->w) + "x" + std::to_string(fpip->h);
		SetWindowText(fp->hwnd, msg.c_str());
	}

	if (fp->check[4] && fp->exfunc->is_saving(fpip->editp))
	{
		// write into buffer
		logdbuf << std::to_string(fpip->frame) << ", " << std::to_string(fpip->w) << ", " << std::to_string(fpip->h) << ", " << std::to_string(msd) << std::endl;
		if (logdbuf.tellp() >= 65536)
		{
			logfiled << logdbuf.str();
			logfiled.flush();
			logdbuf.str(std::string());
		}
	}

	return TRUE;
}

BOOL func_exit_d(FILTER *fp)
{
	logfiled.flush();
	logdbuf.flush();
	logfiled.close();
	//cudaEventDestroy(startd);
	//cudaEventDestroy(stopd);
	//if (dev_DC) errord = cudaFree(dev_DC);
	//if (errord != cudaSuccess)
	//{
	//	MessageBox(fp->hwnd, "cudaFree error on Exit!", "SDeContrast CUDA Error", MB_OK | MB_ICONERROR);
	//	return FALSE;
	//}
	errord = cudaDeviceReset();
	if (errord != cudaSuccess)
	{
		MessageBox(fp->hwnd, "Device cleanup error!", "SDeContrast CUDA Error", MB_OK | MB_ICONERROR);
		return FALSE;
	}
	return TRUE;
}

BOOL func_save_end_d(FILTER *fp, void *editp)
{
	if (fp->check[4])
	{
		logfiled << logdbuf.str();
		logfiled.flush();
		logdbuf.str(std::string());
	}
	return TRUE;
}

BOOL func_update_d(FILTER *fp, int status)
{
	switch (status)
	{
	case FILTER_UPDATE_STATUS_CHECK + 3:
	{
		if (fp->check[3] == 0)
		{
			SetWindowText(fp->hwnd, PLUGIN_NAME_D);
			return FALSE;
		}
	}break;

	}
	return TRUE;
}

BOOL func_WndProc_d(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam, void *editp, FILTER *fp)
{
	switch (message)
	{
	case WM_FILTER_FILE_CLOSE:
	{
		if (dev_DC)
		{
			cudaFree(dev_DC);
			dev_DC = nullptr;
		};
		break;
	}
	};
	return FALSE;
}