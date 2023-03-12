#ifndef __ELAPSED_TIMER_HPP__
#define __ELAPSED_TIMER_HPP__

#include <iostream>
#include <chrono>
#include <cmath>
#include <cassert>
#include <memory>
#include <OnlineStatistics.hpp>
/**
    @brief Timer класс для измерения таймингов между участками кода.
    Поддерживает измерения времени при серийных испытаниях.
*/
template<typename T=double>
class Timer {
public:
private:
    using clock_t=std::chrono::high_resolution_clock;
    using second_t=std::chrono::duration<T, std::ratio<1>>;
    std::chrono::time_point<clock_t> ts_;
    OnlineStatistics<T> stat_;
public:
    Timer() :  ts_(clock_t::now()), stat_{} {};
    /// @brief Засекает начало интервала времени.
    void tic() {
	ts_=clock_t::now();
    };
    /// @brief Засекает конец интервала времени.
    /// @returns длительность интервала времени.
    T  toc() {
	auto dur = std::chrono::duration_cast<second_t>(clock_t::now() - ts_).count();
	stat_.update(dur);
	return dur;
    };
    /// @brief Засекает начало серии испытаний.
    void cleanup() {
	stat_.cleanup();
    };
    /// @brief Функция измерения среднего значение интервала в серии испытаний.
    /// @returns средняя длительность интервала времени.
    T getMean() {
	return stat_.getMean();
    };
    /// @brief Функция измерения стандартного отклонения длительности интервала в серии испытаний.
    /// @returns стандартное отклонение длительности интервала времени.
    T getStd() {
	return stat_.getStd();
    };

};

template<typename T=double>
std::ostream& operator<<(std::ostream& os, Timer<T> timer) {
    auto val=timer.getMean();
    auto std=timer.getStd();
    auto rnd1=[](T v)->T { return std::round(v*10.0)*0.1; };
    if(val<1e-6)		os<<rnd1(val*1e9)<<"±"<<rnd1(std*1e9)<<" ns";
    else if(val<1e-3)		os<<rnd1(val*1e6)<<"±"<<rnd1(std*1e6)<<" µs";
    else if(val<1.0)		os<<rnd1(val*1e3)<<"±"<<rnd1(std*1e3)<<" ms";
    else if(val<60.0)		os<<rnd1(val)<<"±"<<rnd1(std)<<" s";
    else if(val<60.0*60.0) 	os<<rnd1(val/60.0)<<"±"<<rnd1(std/60.0)<<" min";
    else			os<<val/(60.0*60.0)<<"±"<<rnd1(std/(60.0*60.0))<<" hour";
    return os;
};

#endif