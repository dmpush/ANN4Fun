#include <iostream>
#include <string>
#include "DataHolder.hpp"
#include "SimpleTutor.hpp"
#include "Input.hpp"
#include "Layer.hpp"
#include "ANN.hpp"

using namespace std;
int main()
{
    try {
	Input<float> inputs({2});
	Layer<float> layer(&inputs, 3);
	layer.setupTutor(std::make_unique<SimpleTutor<float>>(0.1) );
	layer.setMode(ANN<float>::TrainMode);
	for(int i=0; i<1000; i++) {
	layer.batchBegin();
	inputs.setInput(0, 0.0);
	inputs.setInput(1,-1.0);
	layer.forward();
	layer.setOutput(0, 1.0);
	layer.setOutput(1, 2.0);
	layer.setOutput(2, 3.0);
	layer.backward();

	inputs.setInput(0, 1.0);
	inputs.setInput(1,-0.0);
	layer.forward();
	layer.setOutput(0, 3.14);
	layer.setOutput(1, 2.71);
	layer.setOutput(2, 1.0);
	layer.backward();
	layer.batchEnd();

	layer.setMode(ANN<float>::WorkMode);
	layer.setInput(0, 0.0);
	layer.setInput(1,-1.0);
	layer.forward();
	for(int j=0; j<3; j++)
	    cout<<layer.getOutput(j)<<" ";
	layer.setInput(0, 1.0);
	layer.setInput(1,-0.0);
	layer.forward();
	for(int j=0; j<3; j++)
	    cout<<layer.getOutput(j)<<" ";
	cout<<endl;
    };
    } catch(std::runtime_error ex) {
	cout<<ex.what()<<endl;
    };
    return 0;
};

