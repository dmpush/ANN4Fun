#ifndef __I_DATAHOLDER_HPP__
#define __I_DATAHOLDER_HPP__

#include <random>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <iterator> // std::size
#include <iostream>
#include <algorithm> //std::fill
#include <Tensor.hpp>

/**
    @brief IDataHolder - интерфейс массива памяти для хранения тензоров. 
*/
template<typename T>
class IDataHolder {
public:
    using sPtr=std::shared_ptr<IDataHolder >;
    using uPtr=std::unique_ptr<IDataHolder >;

    IDataHolder(const IDataHolder&) = delete;
    IDataHolder()  {};
    virtual ~IDataHolder() = default;
    /// модификация данных напрямую
    virtual T& raw(size_t ind) = 0;
    /// возвращает указатель на тензор по его имени/ключу
    typename Tensor<T>::sPtr get(std::string name)  {
	auto it=objects_.find(name);
	if(it==objects_.end())
	    throw std::runtime_error(std::string("Нет тензора ")+name+std::string(" в хранилище"));
	return objects_[name];
    };

    /// добавляет в хранилище пару имя тензора/форма тензора
    virtual void append(std::string name, const std::vector<size_t>& dims) = 0;
    /// добавляет пустой тензор 
    virtual void append(std::string name)  = 0;
    /// аллокация памяти хранилища
    virtual void build() = 0;
    /// количество чисел в хранилище
    virtual size_t size() = 0;
    /// создание полной копии хранилища - реализация паттерна Прототип
    virtual sPtr clone()  = 0;
    /// заполнение хранилища константой
    virtual void fill(T val=T(0)) = 0;
    /// печать описания объектов, содержащихся внутри хранилища
    void description() {
	std::cout<<"Размер хранилища "<<size()<<" объектов ("<<size()*sizeof(T)<<" байт)."<<std::endl;
	for(auto [n,o]: objects_) {
	    std::cout<<n<<": ";
	    o->description();
	};
    };
    void dump() {
	for(auto [n,o]: objects_) {
	    std::cout<<n<<"=";
	    o->dump();
	};
    };
    virtual T uniformNoise()  = 0;
    virtual T gaussianNoise() = 0;
    /// true, если хранилище пустое или неинициализированное командой build()
    bool isEmpty() { return size()==0; };

protected:
    void append(std::string name, typename Tensor<T>::sPtr obj) {
        objects_[name]=obj;
    };
    std::map<std::string, typename Tensor<T>::sPtr> objects_;

}; //class
    
#endif
