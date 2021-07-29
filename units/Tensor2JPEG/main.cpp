#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <cmath>
#include <iostream>
#include <IBackendFactory.hpp>
#include <BackendOpenMP.hpp>
#include <Tensor2JPEG.hpp>

using namespace std;

template <typename T>
void test1(typename IBackendFactory<T>::sPtr factory) {
    auto tname=typeid(T).name();
    std::cout<<"Tensor<"<<tname<<"> to JPEG...";
    auto dh=factory->makeHolderS();
    dh->append("X", {640,240});
    dh->build();
    auto X=dh->get("X");
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
void test2(typename IBackendFactory<T>::sPtr factory) {
    auto tname=typeid(T).name();
    std::cout<<"Tensor<"<<tname<<"> to JPEG...";
    auto dh=factory->makeHolderS();
    dh->append("X", {640,240,3});
    dh->build();
    auto X=dh->get("X");
    auto dims=X->dims();
    for(size_t i=0; i<dims[0]; i++)
	for(size_t j=0; j<dims[1]; j++) {
	    float x=static_cast<float>((int)i-(int)dims[0]/2)/static_cast<float>(dims[0]);
	    float y=static_cast<float>((int)j-(int)dims[1]/2)/static_cast<float>(dims[1]);
	    auto A=(x*x+y*y)/0.25;
	    X->val({i,j,0})= (x*x+y*y)*(A<1.0 ? 1.0 : 0.0);
	    X->val({i,j,1})= (x*x+y*y)*(A<0.75 ? 1.0 : 0.0);
	    X->val({i,j,2})= (x*x+y*y)*(A<0.5 ? 1.0 : 0.0);
	};
    tensormath::toJPEG<T>(X, "test-2-"+std::string(tname)+".jpg");
    std::cout<<"ok."<<std::endl;
};

template <typename T>
void test() {
    auto factory=BackendOpenMP<T>::build();
    test1<T>(factory);
    test2<T>(factory);
};
int main()
{
    test<float>();
    test<double>();
    test<long double>();
    std::cout<<"Ok."<<std::endl;
    return 0;
};

