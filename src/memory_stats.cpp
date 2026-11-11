#include "memory_stats.h"

#include <algorithm>
#include <cstdio>
#include <sys/resource.h>

#include <cuda_runtime.h>

MemoryMonitor::MemoryMonitor(int device_id)
    : device_id_(device_id)
{
    sample();
}

void MemoryMonitor::sample()
{
#if defined(__linux__)
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        const size_t rss = static_cast<size_t>(usage.ru_maxrss) * 1024u;
        host_peak_rss_ = std::max(host_peak_rss_, rss);
    }
#endif

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaSetDevice(device_id_) == cudaSuccess &&
        cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        gpu_total_ = total_bytes;
        const size_t used = total_bytes - free_bytes;
        gpu_peak_used_ = std::max(gpu_peak_used_, used);
    }
}

Memory_Usage_Statistics MemoryMonitor::peaks() const
{
    Memory_Usage_Statistics stats;
    stats.gpu_peak_used_bytes = gpu_peak_used_;
    stats.gpu_total_bytes     = gpu_total_;
    stats.host_peak_rss_bytes = host_peak_rss_;
    return stats;
}

void MemoryMonitor::merge_peaks_from(const MemoryMonitor& other)
{
    const Memory_Usage_Statistics other_peaks = other.peaks();
    gpu_peak_used_ = std::max(gpu_peak_used_, other_peaks.gpu_peak_used_bytes);
    if (other_peaks.gpu_total_bytes > 0) {
        gpu_total_ = other_peaks.gpu_total_bytes;
    }
    host_peak_rss_ = std::max(host_peak_rss_, other_peaks.host_peak_rss_bytes);
}

std::string MemoryMonitor::format_bytes(size_t bytes)
{
    static const char* suffix[] = {"B", "KB", "MB", "GB", "TB"};
    double val = static_cast<double>(bytes);
    int idx = 0;
    while (val >= 1024.0 && idx < 4) {
        val /= 1024.0;
        ++idx;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f %s", val, suffix[idx]);
    return std::string(buf);
}
