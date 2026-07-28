#include <cmath>
#include <string>
#include <Spalten/Matrix.hpp>
#include <Spalten/Utils.hpp>
#include "utils.hpp"

void ThreadPool::Start() {
    const int num_threads = std::thread::hardware_concurrency();
    threads.reserve(num_threads);
    for (int i  = 0; i < num_threads; i++) {
        threads.emplace_back(&ThreadPool::ThreadLoop, this);
    }
}

void ThreadPool::ThreadLoop() {
    while (true) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            mutex_condition.wait(lock, [this] {
                return !jobs.empty() || should_terminate;
            });
            if (should_terminate) {
                return;
            }
            job = jobs.front();
            jobs.pop();
        }
        job();

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            pending_tasks--;

            if (pending_tasks == 0) {
                wait_condition.notify_all();
            }
        }
    }
}

void ThreadPool::QueueJob(const std::function<void()>& job) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        jobs.push(job);
        pending_tasks++;
    }
    mutex_condition.notify_one();
}

void ThreadPool::QueueBatch(const std::vector<std::function<void()>>& job_batch) {
    if (job_batch.empty()) return;

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        for (const auto& job : job_batch) {
            jobs.push(job);
        }
        pending_tasks += job_batch.size();
    }
    mutex_condition.notify_all();
}

bool ThreadPool::busy() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    return pending_tasks > 0;
}

void ThreadPool::Wait() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    wait_condition.wait(lock, [this] { return pending_tasks == 0; });
}

void ThreadPool::Stop() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        should_terminate = true;
    }
    mutex_condition.notify_all();
    for (std::thread& active_thread : threads) {
        active_thread.join();
    }
    threads.clear();
}

std::string float_to_string(float f) {
	f = std::round(f * 100.0F) / 100.0F; // Rounding trick: https://stackoverflow.com/a/14369745
    auto f_string = std::to_string(f);
    auto it = f_string.find('.');
    f_string.replace(it, 1, "p");
    return f_string.substr(0, f_string.size() - 3);
}

void warmup() {
    std::cout << "Warming up the CPU...\n";
    Timer t;
    auto A = mat_random_int_range(1000, 1000, 1, 20);
    auto B = mat_random_int_range(1000, 1000, 1, 20);
    volatile auto C = A * B;
    std::cout << "Warmup time: " << t.elapsed() << " seconds." << std::endl;
}