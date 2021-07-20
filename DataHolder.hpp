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
#include <IDataHolder.hpp>

/**
    @brief DataHolder - массив памяти, хранящий в себе именованные переменные - тензоры. В некоторых случаях,
    объекты этого класса интерпретиуются как одномерные тензоры (вектор), например в реализации Учителя.
*/
template<typename T>
class DataHolder : public IDataHolder<T> {
public:
    using sPtr=std::shared_ptr<DataHolder >;
    using uPtr=std::unique_ptr<DataHolder >;

    DataHolder(const DataHolder&) = delete;
    DataHolder() : IDataHolder<T>(), seed_{}, rdev_{seed_()}, uniform_{0.0,1.0}, normal_{0.0, 1.0} {};
    virtual ~DataHolder() = default;
    /// модификация данных напрямую
    T& raw(size_t ind) override { return data_[ind]; }; 
    /// добавляет в хранилище пару имя тензора/форма тензора
    void append(std::string name, const std::vector<size_t>& dims) override {
	auto obj=std::make_shared<Tensor<T>>(this, dims);
	IDataHolder<T>::append(name, obj);
    };
    /// добавляет пустой тензор 
    void append(std::string name) override {
	auto obj=std::make_shared<Tensor<T>>(this);
	IDataHolder<T>::append(name, obj);
    };
    /// аллокация памяти хранилища
    void build() override {
	int offset=0;
	for(auto [name, obj]: IDataHolder<T>::objects_) {
	    obj->setOffset(offset);
	    offset+=obj->size();
	};
	data_.resize(offset);
	std::fill(data_.begin(), data_.end(), T(0));
    };
    /// количество чисел в хранилище
    size_t size() override { return data_.size(); };
    /// создание полной копии хранилища - реализация паттерна Прототип
    typename IDataHolder<T>::sPtr clone() override {
        auto out=std::make_shared<DataHolder<T>>();
	for(auto [name, obj] : IDataHolder<T>::objects_) {
	    auto o=obj->clone();
	    o->holder_=out.get();
	    out->IDataHolder<T>::append(name, o);
	};
	out->build();
	for(size_t i=0; i<size(); i++)
	    out->raw(i)= raw(i);
        return out;
    };
    /// заполнение хранилища константой
    void fill(T val=T(0)) override {
        std::fill(data_.begin(), data_.end(), val);
    };

    T uniformNoise()  override  { return static_cast<T>( uniform_(rdev_) ); };
    T gaussianNoise() override  { return static_cast<T>( normal_ (rdev_) ); };
private:
    std::random_device seed_;
    std::mt19937 rdev_;
    std::uniform_real_distribution<double> uniform_;
    std::normal_distribution<double> normal_;
    std::vector<T> data_;
}; //class
    
#endif
