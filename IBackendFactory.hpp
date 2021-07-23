#ifndef __I_BACKEND_FACTORY__
#define __I_BACKEND_FACTORY__

#include <memory>
#include <IDataHolder.hpp>
#include <ITensor.hpp>

/**
    @brief Интерфейс абстрактной фабрики для смены бэкендов - OpenMP, Cuda etc.
*/

template<typename T>
class IBackendFactory {
public:
    using sPtr=std::shared_ptr<IBackendFactory<T>>;
    IBackendFactory() = default;
    virtual ~IBackendFactory() = default;
//    virtual typename ITensor<T>::sPtr     makeTensorS() = 0;
//    virtual typename ITensor<T>::sPtr     makeTensorU() = 0;
    virtual typename IDataHolder<T>::sPtr makeHolderS() = 0;
    virtual typename IDataHolder<T>::uPtr makeHolderU() = 0;
};

#endif