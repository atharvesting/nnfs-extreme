#pragma once
#include <thread>
#include <functional>
#include <condition_variable>

// Implementation Source: https://stackoverflow.com/a/32593825
class ThreadPool {
public:
    void Start();
    void QueueJob(const std::function<void()>& job);
    void QueueBatch(const std::vector<std::function<void()>>& job_batch);
    void Stop();
    bool busy();

private:
    void ThreadLoop();
    bool should_terminate = false;
    std::mutex queue_mutex;
    std::condition_variable mutex_condition;
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> jobs;
};

std::string float_to_string(float f);
void warmup();

