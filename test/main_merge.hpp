
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>
class ThreadPool {
public:
  ThreadPool(ThreadPool &) = delete;
  // ThreadPool(ThreadPool &&) = delete;
  ThreadPool &operator=(ThreadPool &) = delete;

  ~ThreadPool() { this->stop(); }

  static ThreadPool &instance(const int &num) {
    static ThreadPool ins(num);
    return ins;
  }
  using Task = std::packaged_task<void()>;

  template <class F, class... Args>
  auto commit(F &&f, Args &&...args) -> std::future<decltype(f(args...))> {
    using RetType = decltype(f(args...));
    if (this->stop_.load())
      return std::future<RetType>{};

    auto task = std::make_shared<std::packaged_task<RetType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    {
      std::unique_lock<std::mutex> cv_mt(this->cv_mt_);
      this->tasks_.emplace([task] { (*task)(); });
    }
    std::future<RetType> ret = task->get_future();
    this->cv_lock_.notify_one();
    return ret;
  }

  int idleThreadCount() { return this->thread_num_; }

private:
  ThreadPool(unsigned int num = 5) : stop_(false) {
    if (num < 1)
      this->thread_num_ = 1;
    else
      this->thread_num_ = num;
    start();
  }
  void stop() {
    this->stop_.store(true);
    this->cv_lock_.notify_all();
    for (auto &td : this->pool_) {
      if (td.joinable())
        td.join();
    }
  }
  void start() {
    for (int i = 0; i < this->thread_num_; i++) {
      this->pool_.emplace_back([this]() {
        while (!this->stop_.load()) {
          Task task;
          {
            std::unique_lock<std::mutex> cv_mt(cv_mt_);
            this->cv_lock_.wait(cv_mt, [this] {
              return this->stop_.load() || !this->tasks_.empty();
            });
            if (this->tasks_.empty())
              return;

            task = std::move(this->tasks_.front());
            this->tasks_.pop();
          }
          this->thread_num_--;
          task();
          this->thread_num_++;
        }
      });
    }
  }

  std::mutex cv_mt_;
  std::condition_variable cv_lock_;
  std::queue<Task> tasks_;
  std::vector<std::thread> pool_;
  std::atomic_bool stop_;
  std::atomic_int thread_num_;
};