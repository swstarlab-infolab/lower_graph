#ifndef B325E992_2B0D_49B9_94E7_CFDF007F19E0
#define B325E992_2B0D_49B9_94E7_CFDF007F19E0

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

class WaitGroup
{
private:
	size_t					_original;
	std::atomic<size_t>		_value;
	std::condition_variable _cv;
	std::mutex				_mtx;

public:
	void init(size_t cnt)
	{
		_original = cnt;
		_value.fetch_add(cnt);
	}
	void sync()
	{
		_value.fetch_sub(1);
		if (_value.load() == 0) {
			std::lock_guard<std::mutex> lg(_mtx);
			_value.fetch_add(_original);
			_cv.notify_all();
		} else {
			std::unique_lock<std::mutex> ul(_mtx);
			_cv.wait(ul);
			ul.unlock();
		}
	}
};

#endif /* B325E992_2B0D_49B9_94E7_CFDF007F19E0 */
