#ifndef __ELAPSED_TIMER_HPP__
#define __ELAPSED_TIMER_HPP__

#include <iostream>
#include <chrono>
#include <cmath>

template<typename T=double>
class Timer {
public:
private:
    using clock_t=std::chrono::high_resolution_clock;
    using second_t=std::chrono::duration<T, std::ratio<1>>;
    std::chrono::time_point<clock_t> ts_;
    double mean_, mean2_;
    size_t counter_;
public:
    Timer() :  ts_(clock_t::now()), mean_(0), mean2_(0), counter_(0) {};
    void tic() {
	ts_=clock_t::now();
    };
    T  toc() {
	auto dur = std::chrono::duration_cast<second_t>(clock_t::now() - ts_).count();
	mean_+=dur;
	mean2_+=dur*dur;
	counter_++;
	return dur;
    };
    void cleanup() {
	mean_=0.0;
	mean2_=0.0;
	counter_=0;
    };
    T getMean() {
	return static_cast<T>(mean_)/static_cast<T>(counter_);
    };
    T getStd() {
	double M=static_cast<double>(mean_)/static_cast<double>(counter_);
	double M2=static_cast<double>(mean2_)/static_cast<double>(counter_);
	return static_cast<T>(std::sqrt(std::abs(M*M-M2)));
    };

};



#endif