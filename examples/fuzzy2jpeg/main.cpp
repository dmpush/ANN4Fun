#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>

#include <DataHolder.hpp>
#include <Input.hpp>
#include <Layer.hpp>
#include <Model.hpp>
#include <Dropout.hpp>
#include <ReLUx.hpp>
#include <TestXOR.hpp>
#include <Tensor2JPEG.hpp>
#include <SimpleTutor.hpp>

template<typename T>
class ArctanXOR : public TestXOR<T> {
public:
    std::shared_ptr<Model<T>> buildModel() override {
	std::vector<size_t> inputShape{2};
	std::function<void(T)> validValue=[](T x) { 
	    assert(!std::isnan(x));
	    assert(!std::isinf(x));
	};
	auto model=std::make_shared<Model<T>> (inputShape);
	model-> template addLayer<Layer<T>>({11});
	model-> template addLayer<Dropout<T>>(0.1);
	model-> template addLayer<ReLUx<T>>();

	model-> template addLayer<Layer<T>>({7});
	model-> template addLayer<Dropout<T>>(0.1);
	model-> template addLayer<ReLUx<T>>();

	model-> template addLayer<Layer<T>>({5});
	model-> template addLayer<Dropout<T>>(0.1);
	model-> template addLayer<ReLUx<T>>();

	model-> template addLayer<Layer<T>>({3});
	std::vector<T> regul={0.001};
	model->template setTutor<SimpleTutor<T>>(0.1, regul);
	return model;
    };
    bool assertion() override {
	return TestXOR<T>::getErrorMeanSquare()<0.1;
    };
    void onTunedModel(typename Model<T>::sPtr model) override {
	auto  holder=std::make_shared<DataHolder<T>>();
	holder->append("Img", {256,256,3});
	holder->build();
	auto img=holder->get("Img");
	for(size_t i=0; i<256; i++) {
	    float x=((float)i/256.0)*2.0-1.0;
	    for(size_t j=0; j<256; j++) {
		float y=((float)j/256.0)*2.0-1.0;
		model->setInput(0,(T)x);
		model->setInput(1,(T)y);
		model->forward();
		img->val({i,j,0}) =(T) model->getOutput(0);
		img->val({i,j,1}) =(T) model->getOutput(1);
		img->val({i,j,2}) =(T) model->getOutput(2);
	    };
	};
	tensormath::toJPEG<T>(img, "logic-"+std::to_string(TestXOR<T>::getErrorMax())+".jpg");
    }; 
};



int main()
{
    ArctanXOR<float> test;
    test.enableDropout();
    test.setBatchSize(100);
    test.setNumBatches(1000);
    test.run(10, true);
    return 0;
};

