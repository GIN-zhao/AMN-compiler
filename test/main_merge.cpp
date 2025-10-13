#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

class ThreadPoolMy {

  using Task = std::packaged_task<void()>;

public:
  ThreadPoolMy(const ThreadPoolMy &) = delete;
  ThreadPoolMy(const ThreadPoolMy &&) = delete;
  ThreadPoolMy &operator=(const ThreadPoolMy &) = delete;

  static ThreadPoolMy &instance(const unsigned int &num) {
    static ThreadPoolMy ins(num);
    return ins;
  }

  ~ThreadPoolMy() { stop(); }

  template <typename F, typename... Args>
  auto commit(F &&f, Args &&...args) -> std::future<decltype(f(args...))> {
    using RetType = decltype(f(args...));
    if (stop_.load())
      return std::future<RetType>{};

    auto task = std::make_shared<std::packaged_task<RetType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<RetType> ret = task->get_future();
    {
      std::lock_guard<std::mutex> cv_mt(cv_mt_);
      tasks_.emplace(task);
    }
    cv_lock_.notify_one();
    return ret;
  }

private:
  ThreadPoolMy(unsigned int num) : stop_(false) {
    if (num < 1)
      this->threadnum_ = num;
    else
      this->threadnum_ = num;
    start();
  }

  void start() {
    for (int i = 0; i < this->threadnum_; i++) {
      this->pool_.emplace_back([this]() {
        while (!this->stop_.load()) {
          Task task;
          {
            std::unique_lock<std::mutex> cv_mt(this->cv_mt_);
            this->cv_lock_.wait(cv_mt, [this] {
              return this->stop_.load() || !this->tasks_.empty();
            });
            if (this->tasks_.empty())
              return;
            task = std::move(this->tasks_.front());
            this->tasks_.pop();
          }
          this->threadnum_--;
          task();
          this->threadnum_++;
        }
      });
    }
  }

  void stop() {
    stop_.store(true);
    this->cv_lock_.notify_all();
    for (auto &td : this->pool_) {
      if (td.joinable())
        td.join();
    }
  }

  std::atomic_bool stop_;
  std::mutex cv_mt_;
  std::condition_variable cv_lock_;
  std::atomic_int threadnum_;
  std::queue<Task> tasks_;
  std::vector<std::thread> pool_;
};