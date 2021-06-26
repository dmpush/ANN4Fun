#ifndef __TEST_GAINS_HPP_
#define __TEST_GAINS_HPP_

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
class TestGains : public ModelTest<T> {
public:
    TestGains() : ModelTest<T>() {};
    ~TestGains() = default;

    std::vector<T> getOutput(const std::vector<T> in) override {
	std::vector<T> out(4);
	out[0]= +in[0];
	out[1]= +in[1];
	out[2]= -in[0];
	out[3]= -in[1];
	return out;
    };
    size_t getNumInputs()  override { return 2u; };
    size_t getNumOutputs() override { return 4u; };

private:
};

#endif
