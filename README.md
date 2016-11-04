# SigColorCUDA
Sigmoidal contrast working on RGB colorspace using CUDA 8.0. AviUtl Plugin.

# Requirement
* Nvidia GTX700 series or later
* Latest Nvidia display driver supporting CUDA 8.0
* AviUtl 1.00

# Install
Extract everything into AviUtl's base folder

# Benchmark
On GTX 970, processing a frame of 720P took 2~3ms. 5~6ms for 1080P.

# Patch/Pull request policy
* Please open an issue for discussion first
* Only accept bug fix and major performance improvement tweak
* A mere writing-style change will UNLIKELY be accepted
* Test on your own machine before submitting pull request, don't submit things that does not work

# Related Project
 [CPU-only SigContrast for AviUtl](https://github.com/MaverickTse/SigContrastFastAviUtl)
 
# Building
You will need a compatible CUDA device, VS2015 and CUDA 8.0 installed. Default AviUtl folder for DEBUG use is set to ```C:\AviUtlDBG```. Default CUDA machine support is set for ```compute_35,sm_35;compute_50,sm_50```
