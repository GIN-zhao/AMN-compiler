// #include "threadpool.hpp"
#include "main_merge.hpp"
#include <chrono>
#include <iostream>
#include <vector>

int main() {
  using namespace std::chrono;
  auto &pool = ThreadPool::instance(6);

  const int N = 10;
  std::vector<std::future<int>> futures;

  auto start = high_resolution_clock::now();

  for (int i = 0; i < N; ++i) {
    futures.emplace_back(pool.commit([i]() -> int {
      std::cout << "task " << i << " start on thread "
                << std::this_thread::get_id() << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(2));
      std::cout << "task " << i << " end" << std::endl;
      return i * 10;
    }));
  }

  // collect results
  int sum = 0;
  for (auto &f : futures) {
    try {
      sum += f.get();
    } catch (const std::exception &e) {
      std::cout << "task future exception: " << e.what() << std::endl;
    }
  }

  auto end = high_resolution_clock::now();
  auto ms = duration_cast<milliseconds>(end - start).count();
  std::cout << "sum=" << sum << " elapsed=" << ms << "ms\n";

  std::cout << "idle threads (approx)=" << pool.idleThreadCount() << std::endl;

  return 0;
}
