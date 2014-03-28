#ifndef TIMER_H
#define TIMER_H

#include <time.h>

// measuring time
class Timer
{
    public:
	Timer() : running(false), sec(0)
	{
	}
	void start()
	{
        clock_gettime(CLOCK_MONOTONIC, &tStart);
		running = true;
	}
	void end()
	{
		if (!running) { sec = 0; return; }
        struct timespec tEnd;
        clock_gettime(CLOCK_MONOTONIC, &tEnd);
        sec = (tEnd.tv_sec - tStart.tv_sec) + (tEnd.tv_nsec - tStart.tv_nsec) * 1E-9F;
		running = false;
	}
	float get()
	{
		if (running) end();
		return sec;
	}
    private:
    struct timespec tStart;
	bool running;
	float sec;
};

#endif // TIMER_H
