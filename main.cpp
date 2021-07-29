#include <iostream>
#include <cassert>
#include <string>
#include <BackendOpenMP.hpp>
#include <SimpleTutor.hpp>
#include <Input.hpp>
#include <Layer.hpp>
#include <Model.hpp>
#include <SELU.hpp>
#include <Assertion.hpp>
#include <units/Timer.hpp>
using namespace std;
int main()
{
    Timer timer;
    try {
	Model<float> model({2});
	model.addLayer<Layer<float>>({2});
//	model.template addLayer<Assertion<float>>([](float x){ assert(x<1); });
	model.addLayer<SELU<float>>();
	model.addLayer<Layer<float>>({30});
	model.addLayer<SELU<float>>();
	model.build(BackendOpenMP<float>::build());
//	model(1)->setTutor(std::make_unique<SimpleTutor<float>>(0.1));
//	std::vector<float> L={0.0f,0.00001f};
//	model.setTutor<SimpleTutor<float>>(0.1f, L);
	timer.cleanup();
	for(int i=0; i<1000; i++) {
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

	    timer.tic();
	    model.batchEnd();
	    timer.toc();

	    model.setInput(0, 0.0);
	    model.setInput(1, 1.0);
	    model.forward();
	    for(int j=0; j<3; j++)
		cout<<    model.getOutput(j)<<" ";

	cout<<    model.getInputErrors()->raw(0)<<" ";
	cout<<    model.getInputErrors()->raw(1)<<" ";

	    cout<<" | ";
	    model.setInput(0, 1.0);
	    model.setInput(1, 0.0);
	    model.forward();
	    for(int j=0; j<3; j++)
		cout<<    model.getOutput(j)<<" ";
	cout<<    model.getInputErrors()->raw(0)<<" ";
	cout<<    model.getInputErrors()->raw(1)<<" ";

	cout<<endl;

    };
	cout<<timer<<endl;
    } catch(std::runtime_error& ex) {
	cout<<ex.what()<<endl;
    };
    return 0;
};

