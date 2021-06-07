#ifndef __SUCCESSION_HPP__
#define __SUCCESSION_HPP__

#include <ANN.hpp>

template<typename T>
class Succession : public ANN<T> {
public:
    Succession(ANN<T> *ann) : ANN<T>(ann) {};
    Succession() : ANN<T>() {};
    virtual ~Succession() = default;
private:
};

#endif