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
    virtual T* ref(size_t ind) = 0;
    /// возвращает указатель на тензор по его имени/ключу
    typename ITensor<T>::sPtr get(std::string name)  {
	auto it=objects_.find(name);
	if(it==objects_.end())
	    throw std::runtime_error(std::string("Нет тензора ")+name+std::string(" в хранилище"));
	return objects_[name];
    };

    /// добавляет в хранилище пару имя тензора/форма тензора
    virtual typename ITensor<T>::sPtr append(std::string name, const std::vector<size_t>& dims) = 0;
    /// добавляет пустой тензор 
    virtual typename ITensor<T>::sPtr append(std::string name)  = 0;
    /// аллокация памяти хранилища
    void build() {
	size_t offset=0;
	for(auto [name, obj]: objects_) {
	    setOffset(obj, offset);
	    offset+=obj->size();
	};
	allocate(offset);
	auto obj=append("*", {offset});
	setOffset(obj, 0);
	fill();
    };
    /// количество чисел в хранилище
    virtual size_t size() = 0;
    /// создание полной копии хранилища - реализация паттерна Прототип
    typename IDataHolder<T>::sPtr clone() {
        auto out=makeEmptyObject();
	for(auto [name, obj] : objects_) {
	    if(name != "*" )  {
		auto o=obj->clone();
		setHolder(o, out.get());
		out->append(name, o);
	    }
	};
	out->build();
	for(size_t i=0; i<size(); i++)
	    out->raw(i)= raw(i);
        return out;
    };
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
    /// @brief выделение памяти под тензоры. Зависит от реализации, где находится память.
    virtual void allocate(size_t) = 0;
    /// @brief создает пустой объект. Используется в методе clone()
    virtual typename IDataHolder<T>::sPtr makeEmptyObject() = 0;
    void append(std::string name, typename ITensor<T>::sPtr obj) {
        objects_[name]=obj;
    };
    void setHolder(typename ITensor<T>::sPtr tensor, IDataHolder *holder) {
	tensor->setHolder(holder);
    };

    void setOffset(typename ITensor<T>::sPtr tensor, size_t offset) {
	tensor->setOffset(offset);
    };
    std::map<std::string, typename ITensor<T>::sPtr> objects_;

}; //class
    
#endif
