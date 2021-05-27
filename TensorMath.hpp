#ifndef __TENSOR_MATH_HPP__
#define __TENSOR_MATH_HPP__
/*
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <iterator> // std::size
#include <iostream>
#include <algorithm> //std::fill
*/
#include <DataHolder.hpp>

/**
    Общее правило - последний аргумент функции - результат

*/

namespace tensormath {

template<typename T>
void sum   (typename DataHolder<T>::Tensor::sPtr A,
	    typename DataHolder<T>::Tensor::sPtr B,
	    typename DataHolder<T>::Tensor::sPtr res) {
    if(res->dim()!=A->dim() || res->dim()!=B->dim())
	throw std::runtime_error("sum: Размерности тензоров различны");
    if(res->size()!=A->size() || res->size()!=B->size())
	throw std::runtime_error("sum: Размеры тензоров различны");
    for(size_t i=0; i<res->size(); i++)
	res->set(i, A->get(i)+B->get(i));
};

// res+=A
template<typename T>
void append   (typename DataHolder<T>::Tensor::sPtr A, typename DataHolder<T>::Tensor::sPtr res) {
    if(res->dim()!=A->dim() )
	throw std::runtime_error("append(): Размерности тензоров различны");
    if(res->size()!=A->size() )
	throw std::runtime_error("append(): Размеры тензоров различны");
    for(size_t i=0; i<res->size(); i++)
	res->set(i, A->get(i)+res->get(i));
};




/// Умножение матриц, векторов и матриц, и т.д.
template<typename T>
void mul   (typename DataHolder<T>::Tensor::sPtr A,
	    typename DataHolder<T>::Tensor::sPtr B,
	    typename DataHolder<T>::Tensor::sPtr res) {
	    auto dimsA=A->dims();
	    auto dimsB=B->dims();
//	    std::cout<<A->dim()<<" & "<<B->dim()<<std::endl;
	    /// матрицы разворачиваются строка за строкой
	    if(A->dim()==1 && B->dim()==1) {
		/// скалярное произведение векторов
		if(A->size()!=B->size() || res->size()!=1)
		    throw std::runtime_error("Вектора не сцеплены");
		T s=T(0);
		for(size_t j=0; j<dimsB[1]; j++)
		    s+=A->get(j)*B->get(j);
		res->set(0,s);
	    } else if(A->dim()==1 && B->dim()==2) {
		/// умножение вектора-столбца на матрицу
		if(A->size() != dimsB[1] || dimsB[0]!=res->size() )
		    throw std::runtime_error("Вектор и матрица не сцеплены");
		for(size_t i=0; i<dimsB[0]; i++) {
		    T s=T(0);
		    for(size_t j=0; j<dimsB[1]; j++)
			s+=A->get(j)*B->get({i,j});
		    res->set(i, s);
		};
	    } else if(A->dim()==2 && B->dim()==1) {
		/// умножение матрицы на вектор-столбец
		if(dimsA[0]!=B->size() || dimsA[1]!=res->size())
		    throw std::runtime_error("Матрица и вектор не сцеплены");
		for(size_t i=0; i<dimsA[1]; i++) {
		    T s=T(0);
		    for(size_t j=0; j<dimsA[0]; j++)
			s+=A->get({j,i}) * B->get(j);
		    res->set(i,s);
		};
	    } else {
		throw  std::runtime_error("Умножение не релизовано");
	    };
};

template<typename T>
void copy  (typename DataHolder<T>::Tensor::sPtr src,
	    typename DataHolder<T>::Tensor::sPtr dest) {
    if(src->size()!=dest->size())
	throw  std::runtime_error("copy(): тензоры имеют разные размеры");
    for(size_t i=0; i<src->size(); i++)
	dest->set(i, src->get(i));
};
/// внешнее произведение двух векторов
template<typename T>
void extmulapp(typename DataHolder<T>::Tensor::sPtr A,
	    typename DataHolder<T>::Tensor::sPtr B,
	    typename DataHolder<T>::Tensor::sPtr res) {
    if(A->dim()!=1 && B->dim()!=1 || res->dim()!=2) 
	throw std::runtime_error("extmul(): входные тензоры должны быть векторами, а выходной - матрицей");
    auto dimsA=A->dims();
    auto dimsB=B->dims();
    auto dimsC=res->dims();
    
    if(dimsA[0]!=dimsC[0] || dimsB[0] != dimsC[1])
	throw std::runtime_error("extmulapp(): входные тензоны не согласованны с выходным");
    for(size_t i=0; i<dimsC[0]; i++)
	for(size_t j=0; j<dimsC[1]; j++)
	    res->set({i,j}, res->get({i,j}) + A->get(i) * B->get(j) );
};

};//namespace
#endif