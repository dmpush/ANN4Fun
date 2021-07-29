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
#include <ITensor.hpp>
#include <IDataHolder.hpp>

/**
    @brief DataHolder - массив памяти, хранящий в себе именованные переменные - тензоры. 
    Основная особенность - данные хранятся в обычной памяти компьютера.
*/
template<typename T, typename TensorType>
class DataHolder : public IDataHolder<T> {
public:
    DataHolder(const DataHolder&) = delete;
    DataHolder() : IDataHolder<T>(), seed_{}, rdev_{seed_()}, uniform_{0.0,1.0}, normal_{0.0, 1.0} {};
    virtual ~DataHolder() = default;
    /// модификация данных напрямую
    T& raw(size_t ind) override { return data_[ind]; }; 
    T* ref(size_t ind) override { return &data_[ind]; }; 
    /// добавляет в хранилище пару имя тензора/форма тензора
    typename ITensor<T>::sPtr append(std::string name, const std::vector<size_t>& dims) override {
	auto obj=std::make_shared<TensorType>(this, dims);
	IDataHolder<T>::append(name, obj);
	return obj;
    };
    /// добавляет пустой тензор 
    typename ITensor<T>::sPtr append(std::string name) override {
	auto obj=std::make_shared<TensorType>(this);
	IDataHolder<T>::append(name, obj);
	return obj;
    };

    /// количество чисел в хранилище
    size_t size() override { return data_.size(); };

    /// заполнение хранилища константой
    void fill(T val=T(0)) override {
        std::fill(data_.begin(), data_.end(), val);
    };

    T uniformNoise()  override  { return static_cast<T>( uniform_(rdev_) ); };
    T gaussianNoise() override  { return static_cast<T>( normal_ (rdev_) ); };

protected:
void allocate(size_t size) override {
	data_.resize(size);
};
typename IDataHolder<T>::sPtr makeEmptyObject() override {
    return std::make_shared<DataHolder<T,TensorType>>();
};
private:
    std::random_device seed_;
    std::mt19937 rdev_;
    std::uniform_real_distribution<double> uniform_;
    std::normal_distribution<double> normal_;
    std::vector<T> data_;
}; //class
    
#endif
