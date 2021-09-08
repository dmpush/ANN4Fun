#ifndef __SUCCESSION_HPP__
#define __SUCCESSION_HPP__

#include <ANN.hpp>

/**
    @brief Succession - абстрактный класс сетей, являющихся строительными блоками. Конструкторы таких 
	сетей должны получать в качестве обязательного параметра указатель на другую сеть (слой или подсеть).
*/
template<typename T>
class Succession : public ANN<T> {
public:
    Succession(typename ANN<T>::sPtr ann) : ANN<T>(ann) {};
    Succession() : ANN<T>() {};
    virtual ~Succession() = default;
private:
};

#endif