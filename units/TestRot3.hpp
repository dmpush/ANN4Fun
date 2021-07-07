#ifndef __TEST_ROT3_HPP_
#define __TEST_ROT3_HPP_

#include <iostream>
#include <string>
#include <cassert>
#include <cmath> //abs
#include <cstdlib> //rand
#include <algorithm> //min,max
#include <memory>
#include <array>

#include <DataHolder.hpp>
#include <Model.hpp>
#include <ModelTest.hpp>

using namespace std;

template<typename T>
class TestRot3: public ModelTest<T> {
    std::array<std::array<double,3>,3> M;
public:
    TestRot3() : ModelTest<T>() { 
	// генерация трех углов Эйлера
	double a = ModelTest<T>::uniformNoise()*std::numbers::pi*2.0;
	double b = ModelTest<T>::uniformNoise()*std::numbers::pi*2.0;
	double c = ModelTest<T>::uniformNoise()*std::numbers::pi*2.0;
	double  cosA, sinA;
	double  cosB, sinB;
	double  cosC, sinC;
	cosA=std::cos(a); sinA=std::sin(a);
	cosB=std::cos(b); sinB=std::sin(b);
	cosC=std::cos(c); sinC=std::sin(c);
	// матрица поворота
	M[0][0] = cosA*cosC - sinA*cosB*sinC;
	M[0][1] = sinA*cosC + cosA*cosB*sinC;
	M[0][2] = sinB*sinC;
	M[1][0] = -cosA*sinC - sinA*cosB*cosC;
	M[1][1] = -sinA*sinC + cosA*cosB*cosC;
	M[1][2] = sinB*cosC;
	M[2][0] = sinA*sinB;
	M[2][1] = -cosA*sinB;
	M[2][2]= cosB;
    };
    ~TestRot3() = default;

    std::vector<T> getInput() override {
	std::vector<T> out(3);
	for(auto& o : out)
	    o= ModelTest<T>::uniformNoise()*2.0 - 1.0;
	return out;
    };

    std::vector<T> getOutput(const std::vector<T> in) override {
	std::vector<T> out(3);
	out[0]=M[0][0]*in[0] + M[0][1]*in[1] + M[0][2]*in[2];
	out[1]=M[1][0]*in[0] + M[1][1]*in[1] + M[1][2]*in[2];
	out[2]=M[2][0]*in[0] + M[2][1]*in[1] + M[2][2]*in[2];
	return out;
    };
    size_t getNumInputs()  override { return 3u; };
    size_t getNumOutputs() override { return 3u; };
private:
};

#endif
