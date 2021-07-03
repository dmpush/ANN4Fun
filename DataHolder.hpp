#ifndef DATAHOLDER_HPP
#define DATAHOLDER_HPP

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
    @brief DataHolder - массив памяти, хранящий в себе именованные переменные - тензоры. В некоторых случаях,
    объекты этого класса интерпретиуются как одномерные тензоры (вектор), например в реализации Учителя.
*/
template<typename T>
class DataHolder {
public:
    using sPtr=std::shared_ptr<DataHolder >;
    using uPtr=std::unique_ptr<DataHolder >;

    DataHolder(const DataHolder&) = delete;
    DataHolder() : seed_{}, rdev_{seed_()}, uniform_{0.0,1.0}, normal_{0.0, 1.0} {};
    virtual ~DataHolder() = default;
    /// модификация данных напрямую
    T& raw(size_t ind) { return data_[ind]; }; 
    /// возвращает указатель на тензор по его имени/ключу
    typename Tensor<T>::sPtr get(std::string name)  {
	auto it=objects_.find(name);
	if(it==objects_.end())
	    throw std::runtime_error(std::string("Нет тензора ")+name+std::string(" в хранилище"));
	return objects_[name];
    };

    /// добавляет в хранилище пару имя тензора/форма тензора
    void append(std::string name, const std::vector<size_t>& dims) {
	auto obj=std::make_shared<Tensor<T>>(this, dims);
	append(name, obj);
    };
    /// добавляет пустой тензор 
    void append(std::string name) {
	auto obj=std::make_shared<Tensor<T>>(this);
	append(name, obj);
    };
    /// аллокация памяти хранилища
    void build() {
	int offset=0;
	for(auto [name, obj]: objects_) {
	    obj->setOffset(offset);
	    offset+=obj->size();
	};
	data_.resize(offset);
	std::fill(data_.begin(), data_.end(), T(0));
    };
    /// количество чисел в хранилище
    size_t size() { return data_.size(); };
    /// создание полной копии хранилища
    void clone(typename DataHolder<T>::sPtr src) {
	for(auto [name, obj] : src->objects_) {
	    auto o=obj->clone();
	    o->holder_=this;
	    append(name, o);
	};
	build();
	for(size_t i=0; i<size(); i++)
	    raw(i)= src->raw(i);
    };
    /// заполнение хранилища константой
    void fill(T val=T(0)) {
        std::fill(data_.begin(), data_.end(), val);
    };
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
    T uniformNoise()  { return static_cast<T>( uniform_(rdev_) ); };
    T gaussianNoise() { return static_cast<T>( normal_ (rdev_) ); };
    /// true, если хранилище пустое или неинициализированное командой build()
    bool isEmpty() { return size()==0; };
private:
    void append(std::string name, typename Tensor<T>::sPtr obj) {
        objects_[name]=obj;
    };

    std::random_device seed_;
    std::mt19937 rdev_;
    std::uniform_real_distribution<double> uniform_;
    std::normal_distribution<double> normal_;
    std::vector<T> data_;
    std::map<std::string, typename Tensor<T>::sPtr> objects_;

}; //class
    
#endif
