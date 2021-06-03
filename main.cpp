#include <iostream>
#include <string>
#include <DataHolder.hpp>
#include <SimpleTutor.hpp>
#include <Input.hpp>
#include <Layer.hpp>
#include <Model.hpp>

using namespace std;
int main()
{
    try {
	Model<float> model({2});
	model.addLayer<Layer<float>>({2});
	model.addLayer<Layer<float>>({3});

	model(1)->setTutor(std::make_unique<SimpleTutor<float>>(0.1));

	for(int i=0; i<1000; i++) {
	    model.setMode(ANN<float>::TrainMode);

	    model.batchBegin();

	    model.setInput(0, 0.0);
	    model.setInput(1, 1.0);
	    model.forward();
	    model.setOutput(0, 1.0);
	    model.setOutput(1, 2.0);
	    model.setOutput(2, 3.0);
	    model.backward();

	    model.setInput(0, 1.0);
	    model.setInput(1, 0.0);
	    model.forward();
	    model.setOutput(0, 3.14);
	    model.setOutput(1, 2.71);
	    model.setOutput(2, 1.41);
	    model.backward();

	    model.batchEnd();


	    model.setMode(ANN<float>::WorkMode);
	    model.setInput(0, 0.0);
	    model.setInput(1, 1.0);
	    model.forward();
	    for(int j=0; j<3; j++)
		cout<<    model.getOutput(j)<<" ";

	cout<<    model.getInputErrors()->get(0)<<" ";
	cout<<    model.getInputErrors()->get(1)<<" ";

	    cout<<" | ";
	    model.setInput(0, 1.0);
	    model.setInput(1, 0.0);
	    model.forward();
	    for(int j=0; j<3; j++)
		cout<<    model.getOutput(j)<<" ";
	cout<<    model.getInputErrors()->get(0)<<" ";
	cout<<    model.getInputErrors()->get(1)<<" ";

	cout<<endl;
    };
    } catch(std::runtime_error ex) {
	cout<<ex.what()<<endl;
    };
    return 0;
};

