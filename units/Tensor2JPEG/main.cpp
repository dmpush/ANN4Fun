#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <cmath>
#include <iostream>
#include <DataHolder.hpp>
#include <Tensor2JPEG.hpp>

using namespace std;

template <typename T>
void test1() {
    auto tname=typeid(T).name();
    std::cout<<"Tensor<"<<tname<<"> to JPEG...";
    DataHolder<T> dh;
    dh.append("X", {640,240});
    dh.build();
    auto X=dh.get("X");
    auto dims=X->dims();
    for(size_t i=0; i<dims[0]; i++)
	for(size_t j=0; j<dims[1]; j++) {
	    float x=static_cast<float>((int)i-(int)dims[0]/2)/static_cast<float>(dims[0]);
	    float y=static_cast<float>((int)j-(int)dims[1]/2)/static_cast<float>(dims[1]);
	    auto A=(x*x+y*y)/0.25;
	    X->val({i,j})= (x*x+y*y)*(A<1.0 ? 1.0 : 0.0);
	};
    tensormath::toJPEG<T>(X, "test-"+std::string(tname)+".jpg");
    std::cout<<"ok."<<std::endl;
};

template <typename T>
void test() {
    test1<T>();
};
int main()
{
    test<float>();
    test<double>();
    test<long double>();
    std::cout<<"Ok."<<std::endl;
    return 0;
};

