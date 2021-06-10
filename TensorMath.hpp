#ifndef __TENSOR_MATH_HPP__
#define __TENSOR_MATH_HPP__

#include <DataHolder.hpp>

/**
    @file TensorMath.hpp Общее правило - последний аргумент функции - результат

*/

namespace tensormath {

/// @brief sum() - поэлементная сумма двух тензоров с записью в третий тензор: res=A+B
/// @param A,B - слагаемые
/// @param res - сумма
template<typename T>
void sum   (Tensor<T> A, Tensor<T> B, Tensor<T> res) {
    if(res->dim()!=A->dim() || res->dim()!=B->dim())
	throw std::runtime_error("sum: Размерности тензоров различны");
    if(res->size()!=A->size() || res->size()!=B->size())
	throw std::runtime_error("sum: Размеры тензоров различны");
    for(size_t i=0; i<res->size(); i++)
	res->raw(i) = A->raw(i) + B->raw(i);
};

/// @brief append() - операция += для тензоров: res+=A
template<typename T>
void append   (Tensor<T> A, Tensor<T> res) {
    if(res->dim()!=A->dim() )
	throw std::runtime_error("append(): Размерности тензоров различны");
    if(res->size()!=A->size() )
	throw std::runtime_error("append(): Размеры тензоров различны");
    for(size_t i=0; i<res->size(); i++)
	res->raw(i) = res->raw(i) + A->raw(i);
};




///  @brief mul() -  Умножение матриц, векторов и матриц, и т.д.: res=A*B
template<typename T>
void mul   (Tensor<T> A, Tensor<T> B, Tensor<T> res) {
	    auto dimsA=A->dims();
	    auto dimsB=B->dims();

	    // матрицы разворачиваются строка за строкой
	    if(A->dim()==1 && B->dim()==1) {
		// скалярное произведение векторов
		if(A->size()!=B->size() || res->size()!=1)
		    throw std::runtime_error("Вектора не сцеплены");
		T s{0};
		for(size_t j=0; j<dimsB[1]; j++)
		    s+=A->raw(j)*B->raw(j);
		res->raw(0) = s;
	    } else if(A->dim()==1 && B->dim()==2) {
		// умножение вектора-столбца на матрицу
		if(A->size() != dimsB[1] || dimsB[0]!=res->size() )
		    throw std::runtime_error("Вектор и матрица не сцеплены");
		for(size_t i=0; i<dimsB[0]; i++) {
		    T s{0};
		    for(size_t j=0; j<dimsB[1]; j++)
			s+=A->raw(j)*B->val({i,j});
		    res->raw(i) = s;
		};
	    } else if(A->dim()==2 && B->dim()==1) {
		// умножение матрицы на вектор-столбец
		if(dimsA[0]!=B->size() || dimsA[1]!=res->size())
		    throw std::runtime_error("Матрица и вектор не сцеплены");
		for(size_t i=0; i<dimsA[1]; i++) {
		    T s{0};
		    for(size_t j=0; j<dimsA[0]; j++)
			s+=A->val({j,i}) * B->raw(j);
		    res->raw(i) = s;
		};
	    } else {
		throw  std::runtime_error("Умножение не релизовано");
	    };
};
/// @brief copy() - название говорит само за себя. Копирование тензоров: dest=src
template<typename T>
void copy  (Tensor<T> src, Tensor<T> dest) {
    if(src->size()!=dest->size())
	throw  std::runtime_error("copy(): тензоры имеют разные размеры");
    for(size_t i=0; i<src->size(); i++)
	dest->raw(i) = src->raw(i);
};
/// @brief extmulapp() внешнее произведение двух векторов c добавлением к двухмерной матрице: res+=A*B
template<typename T>
void extmulapp(Tensor<T> A,Tensor<T> B, Tensor<T> res) {
    if(A->dim()!=1 && B->dim()!=1 || res->dim()!=2) 
	throw std::runtime_error("extmul(): входные тензоры должны быть векторами, а выходной - матрицей");
    auto dimsA=A->dims();
    auto dimsB=B->dims();
    auto dimsC=res->dims();
    
    if(dimsA[0]!=dimsC[0] || dimsB[0] != dimsC[1])
	throw std::runtime_error("extmulapp(): входные тензоны не согласованны с выходным");
    for(size_t i=0; i<dimsC[0]; i++)
	for(size_t j=0; j<dimsC[1]; j++)
	    res->val({i,j}) = res->val({i,j}) + A->raw(i) * B->raw(j);
};

};//namespace
#endif