#pragma once
#include <queue>
#include <thread>
#include <functional>
#include <condition_variable>

// Implementation Source: https://stackoverflow.com/a/32593825
class ThreadPool {
public:
    ~ThreadPool();
    void Start();
    void QueueJob(const std::function<void()>& job);
    void QueueBatch(const std::vector<std::function<void()>>& job_batch);
    void Wait();
    void Stop();
    bool busy();

private:
    void ThreadLoop();
    
    bool should_terminate = false;
    size_t pending_tasks = 0;
    std::mutex queue_mutex;
    std::condition_variable mutex_condition;
    std::condition_variable wait_condition;
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> jobs;
};

std::string float_to_string(float f);
void warmup();

// End is inclusive in each range!
template <typename T>
std::vector<std::pair<int, int>> create_ranges(std::vector<T>& container, int thread_count) {
    size_t container_size = container.size();

    std::vector<std::pair<int, int>> ranges;
    ranges.reserve(thread_count);

    size_t chunk_size = container_size / thread_count;
    size_t remainder = container_size % thread_count;
    size_t start = 0;
    for (int i = 0; i < thread_count; i++) 
    {
        size_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);
        if (current_chunk_size == 0) break;

        size_t end = start + current_chunk_size - 1;
        ranges.push_back({ static_cast<int>(start), static_cast<int>(end) });
        start += current_chunk_size;
    }
    return ranges;
}

