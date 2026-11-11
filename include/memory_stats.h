#ifndef MEMORY_STATS_H
#define MEMORY_STATS_H

#include <cstddef>
#include <string>

struct Memory_Usage_Statistics
{
    size_t gpu_peak_used_bytes = 0;
    size_t gpu_total_bytes     = 0;
    size_t host_peak_rss_bytes = 0;
};

class MemoryMonitor
{
public:
    explicit MemoryMonitor(int device_id = 0);

    void sample();
    Memory_Usage_Statistics peaks() const;
    void merge_peaks_from(const MemoryMonitor& other);

    static std::string format_bytes(size_t bytes);

private:
    int device_id_;
    size_t gpu_peak_used_ = 0;
    size_t gpu_total_     = 0;
    size_t host_peak_rss_ = 0;
};

#endif // MEMORY_STATS_H
