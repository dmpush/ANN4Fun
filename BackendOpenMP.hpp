#ifndef __BACKEND_OPENMP__
#define __BACKEND_OPENMP__

#include <memory>
#include <IBackendFactory.hpp>
#include <DataHolder.hpp>
#include <Tensor.hpp>

/**
    @brief Фабрика для бэкенда OpenMP.
*/

template<typename T>
class BackendOpenMP: public IBackendFactory<T> {
public:
    BackendOpenMP() : IBackendFactory<T>() {};
    ~BackendOpenMP() {};
/*    typename ITensor<T>::sPtr makeTensorS() override final {
	return std::make_shared<Tensor<T>>();
    };
    typename ITensor<T>::uPtr makeTensorU() override final {
	return std::move(std::make_unique<Tensor<T>>());
    };
*/
    typename IDataHolder<T>::sPtr makeHolderS() override final {
	return std::make_shared<DataHolder<T>>();
    };
    typename IDataHolder<T>::uPtr makeHolderU() override final {
	return std::make_unique<DataHolder<T>>();
    };
    static typename IBackendFactory<T>::sPtr build() {
	return std::make_shared<BackendOpenMP>();
    };
};

#endif