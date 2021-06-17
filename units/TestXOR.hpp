#ifndef __TEST_XOR_HPP_
#define __TEST_XOR_HPP_

#include <iostream>
#include <string>
#include <cassert>
#include <cmath> //abs
#include <cstdlib> //rand
#include <algorithm> //min,max
#include <memory>

#include <DataHolder.hpp>
#include <Model.hpp>
#include <ModelTest.hpp>

using namespace std;

template<typename T>
class TestXOR : public ModelTest<T> {
public:
    TestXOR() : scale_(0.9), offset_(0.0), ModelTest<T>() {};
    ~TestXOR() = default;

    virtual std::vector<T> getOutput(const std::vector<T> in) override {
	std::vector<T> out(3);
	out[0]= AND(in[0], in[1])*scale_+offset_;
	out[1]= OR (in[0], in[1])*scale_+offset_;
	out[2]= XOR(in[0], in[1])*scale_+offset_;
	return out;
    };

private:
    T scale_, offset_;
    T NOT(T x) { return  - x; };
    T OR(T x, T y) { return std::max(x,y); };
    T AND(T x, T y) { return std::min(x,y); };
    T XOR(T x, T y) { return OR(AND(x, NOT(y)), AND(NOT(x), y)) ; };
};

#endif
