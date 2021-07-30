#ifndef __TENSOR_OMP_HPP__
#define __TENSOR_OMP_HPP__

#include <cmath>
#include <vector>
#include <ITensor.hpp>
//#include <IDataHolder.hpp>

template<typename T>
class IDataHolder;


/**
    @brief Tensor - класс, реализующий многомерный массив (тензор). 
*/
template<typename T>
class TensorOMP : public ITensor<T> {
public:
    friend class IDataHolder<T>;

    TensorOMP() = delete;
    explicit TensorOMP(IDataHolder<T> *holder,  const std::vector<size_t>& dimensions) : ITensor<T>(holder, dimensions) {
    };

    explicit TensorOMP(IDataHolder<T> *holder) : ITensor<T>(holder) {};

    TensorOMP(const TensorOMP&) = delete;
    ~TensorOMP()=default;
    /// паттерн Прототип
    typename ITensor<T>::sPtr clone() override {
	auto out=std::make_shared<TensorOMP>(nullptr, ITensor<T>::dims());
	out->setOffset(ITensor<T>::getOffset());
	return out;
    };

public:
/// @brief sum() - поэлементная сумма двух тензоров с записью в третий тензор: this=A+B
/// @param A,B - слагаемые
void sum   (typename ITensor<T>::sPtr A, typename ITensor<T>::sPtr B) override {
    if(ITensor<T>::dim()!=A->dim() || ITensor<T>::dim()!=B->dim())
	throw std::runtime_error("sum: Размерности тензоров различны");
    if(ITensor<T>::size()!=A->size() || ITensor<T>::size()!=B->size())
	throw std::runtime_error("sum: Размеры тензоров различны");
    for(size_t i=0; i<ITensor<T>::size(); i++)
	ITensor<T>::raw(i) = A->raw(i) + B->raw(i);
};
/// @brief prod() - поэлементное произведение  двух тензоров с записью в третий тензор: this=A.*B
/// @param A,B - сомножители
void prod   (typename ITensor<T>::sPtr A, typename ITensor<T>::sPtr B) override {
    if(ITensor<T>::dim()!=A->dim() || ITensor<T>::dim()!=B->dim())
	throw std::runtime_error("sum: Размерности тензоров различны");
    if(ITensor<T>::size()!=A->size() || ITensor<T>::size()!=B->size())
	throw std::runtime_error("sum: Размеры тензоров различны");
    for(size_t i=0; i<ITensor<T>::size(); i++)
	ITensor<T>::raw(i) = A->raw(i) * B->raw(i);
};

/// @brief prodapp() - поэлементное произведение  двух тензоров с суммированием с  третим тензором: this+=A.*B
/// @param A,B - сомножители
void prodapp(typename ITensor<T>::sPtr A, typename ITensor<T>::sPtr B) override {
    if(ITensor<T>::dim()!=A->dim() || ITensor<T>::dim()!=B->dim())
	throw std::runtime_error("sum: Размерности тензоров различны");
    if(ITensor<T>::size()!=A->size() || ITensor<T>::size()!=B->size())
	throw std::runtime_error("sum: Размеры тензоров различны");
    for(size_t i=0; i<ITensor<T>::size(); i++)
	ITensor<T>::raw(i) = ITensor<T>::raw(i) + A->raw(i) * B->raw(i);
};


/// @brief append() - операция += для тензоров: this+=A
/// @param A -- аргумент
void append   (typename ITensor<T>::sPtr A) override {
    if(ITensor<T>::dim()!=A->dim() )
	throw std::runtime_error("append(): Размерности тензоров различны");
    if(ITensor<T>::size()!=A->size() )
	throw std::runtime_error("append(): Размеры тензоров различны");
    #pragma omp parallel for
    for(size_t i=0; i<ITensor<T>::size(); i++)
	ITensor<T>::raw(i) = ITensor<T>::raw(i) + A->raw(i);
};

///  @brief mul() -  Умножение матриц, векторов и матриц, и т.д.: this=A*B
/// @param A,B -- сомножители
void mul   (typename ITensor<T>::sPtr A, typename ITensor<T>::sPtr B) override {
	    auto dimsA=A->dims();
	    auto dimsB=B->dims();

	    // матрицы разворачиваются строка за строкой
	    if(A->dim()==1 && B->dim()==1) {
		// скалярное произведение векторов
		if(A->size()!=B->size() || ITensor<T>::size()!=1)
		    throw std::runtime_error("Вектора не сцеплены");
		T s{0};
		for(size_t j=0; j<dimsB[1]; j++)
		    s+=A->raw(j)*B->raw(j);
		ITensor<T>::raw(0) = s;
	    } else if(A->dim()==1 && B->dim()==2) {
		// умножение вектора-столбца на матрицу
		if(A->size() != dimsB[1] || dimsB[0]!=ITensor<T>::size() )
		    throw std::runtime_error("Вектор и матрица не сцеплены");
		#pragma omp parallel for
		for(size_t i=0; i<dimsB[0]; i++) {
		    T s{0};
		    for(size_t j=0; j<dimsB[1]; j++)
			s+=A->raw(j)*B->val(i,j);
		    ITensor<T>::raw(i) = s;
		};
	    } else if(A->dim()==2 && B->dim()==1) {
		// умножение матрицы на вектор-столбец
		if(dimsA[0]!=B->size() || dimsA[1]!=ITensor<T>::size())
		    throw std::runtime_error("Матрица и вектор не сцеплены");
		#pragma omp parallel for
		for(size_t i=0; i<dimsA[1]; i++) {
		    T s{0};
		    for(size_t j=0; j<dimsA[0]; j++)
			s+=A->val(j,i) * B->raw(j);
		    ITensor<T>::raw(i) = s;
		};
	    } else {
		throw  std::runtime_error("Умножение не релизовано");
	    };
};

/// @brief copy() -- название говорит само за себя. Копирование тензоров. this=src
/// @param src -- источник
void copy  (typename ITensor<T>::sPtr src) override {
    if(src->size()!=ITensor<T>::size())
	throw  std::runtime_error("copy(): тензоры имеют разные размеры");
    #pragma omp parallel for
    for(size_t i=0; i<src->size(); i++)
	ITensor<T>::raw(i) = src->raw(i);
};

/// @brief fill() -- заполнение тензора постоянным значением val
/// @param val -- заполнитель тензора
void fill  (T val) override {
    #pragma omp parallel for
    for(size_t i=0; i<ITensor<T>::size(); i++)
	ITensor<T>::raw(i) = val;
};

/// @brief extmulapp() внешнее произведение двух векторов c добавлением к двухмерной матрице: this+=A*B
/// @param A,B -- сомножители
void extmulapp(typename ITensor<T>::sPtr A, typename ITensor<T>::sPtr B) override {
    if(A->dim()!=1 || B->dim()!=1 || ITensor<T>::dim()!=2) 
	throw std::runtime_error("extmul(): входные тензоры должны быть векторами, а выходной - матрицей");
    auto dimsA=A->dims();
    auto dimsB=B->dims();
    auto dimsC=ITensor<T>::dims();
    
    if(dimsA[0]!=dimsC[0] || dimsB[0] != dimsC[1])
	throw std::runtime_error("extmulapp(): входные тензоны не согласованны с выходным");
    #pragma omp parallel for
    for(size_t i=0; i<dimsC[0]; i++)
	for(size_t j=0; j<dimsC[1]; j++)
	    ITensor<T>::val({i,j}) = ITensor<T>::val({i,j}) + A->raw(i) * B->raw(j);
};

/// @brief gaussianNoise - заполнение тензора шумом Гаусса
/// @param M -- математическое ожидание
/// @param S -- среднеквадатичное отклонение
void gaussianNoise(T M, T S) override {
    auto holder=ITensor<T>::getHolder();
    for(size_t i=0; i<ITensor<T>::size(); i++)
	ITensor<T>::raw(i) = holder->gaussianNoise()*S + M; ;
    };

/// @brief uniformNoise - заполнение тензора шумом Гаусса
/// @param a -- нижняя граница значений шума
/// @param b -- верхняя граница значений шума
void uniformNoise(double a, double b) override {
    auto holder=ITensor<T>::getHolder();
    for(size_t i=0; i<ITensor<T>::size(); i++)
	ITensor<T>::raw(i) = holder->uniformNoise() * (b-a)+a;
    };
/// @brief распределение Бернулли
/// @param prob -- вероятнось выпадания орла (heads) либо решки (tails)
void bernoulliNoise(double prob, T heads=1.0, T tails=0.0) override {
    auto holder=ITensor<T>::getHolder();
    for(size_t i=0; i<ITensor<T>::size(); i++)
	ITensor<T>::raw(i) = holder->uniformNoise() < prob ? heads : tails;
    };
/// @brief apply -- применение функции ко всем элементам тензора
/// @param func -- функция
void apply(const std::function<T(T)>&func) override {
    for(size_t i=0; i<ITensor<T>::size(); i++)
	ITensor<T>::raw(i) = func(ITensor<T>::raw(i));
};
/// @brief оптимизация методом градиентного спуска с постоянным шагом.
/// @param grad -- тензор с градиентом
/// @param scale -- коэфф-т к градиенту - число, обратное к числу сэмплов в батче
/// @param step -- шаг градиентного метода
/// @param regpoly -- массив коэфф-тов полином регуляризации
void optGrad(typename ITensor<T>::sPtr grad, T scale, T step, const std::vector<T>& regpoly)  override {
    #pragma omp parallel for
    for(size_t i=0; i<ITensor<T>::size(); i++)  {
	auto X=ITensor<T>::raw(i);
	auto dX=grad->raw(i)*scale - this->getRegularization(regpoly, X);
	ITensor<T>::raw(i) = X  + step *  dX ;
    };
};
/// @brief оптимизация методом Нестерова
/// @param grad -- тензор с градиентом
/// @param velocity -- тензор со сглаженным граентом
/// @param scale -- коэфф-т к градиенту - число, обратное к числу сэмплов в батче
/// @param step -- шаг градиентного метода
/// @param beta -- коэфф-т инерции (beta~0 - градиентный метод)
/// @param regpoly -- массив коэфф-тов полином регуляризации
void optNesterov(typename ITensor<T>::sPtr grad, typename ITensor<T>::sPtr velocity, 
	T scale, T step, T beta, const std::vector<T>& regpoly) override {
    #pragma omp parallel for
    for(size_t i=0; i<ITensor<T>::size(); i++)  {
	auto X=this->raw(i);
	auto dX=grad->raw(i)*scale - this->getRegularization(regpoly, X);
	auto V=velocity->raw(i);
	V=V*beta + dX*(1.0 - beta);
	this->raw(i) = X + step * V;
	velocity->raw(i) = V;
    };
};
///
void optAdam(typename ITensor<T>::sPtr grad, typename ITensor<T>::sPtr Mt, typename ITensor<T>::sPtr Vt,
	T scale, T batchCount, 
	T dt, T beta1, T beta2, T eps,
	const std::vector<T>& regpoly) override {
    T Beta1=std::pow(beta1, batchCount+1.0);
    T Beta2=std::pow(beta2, batchCount+1.0);
    #pragma omp parallel for
    for(size_t i=0; i<this->size(); i++)  {
	auto X=this->raw(i);
	auto dX=grad->raw(i)*scale - this->getRegularization(regpoly, X);
	auto M=Mt->raw(i);
	auto V=Vt->raw(i);
	M=M*beta1 + (1.0-beta1) * dX;
	V=V*beta2 + (1.0-beta2) * dX*dX;
	auto Mx=M/(1.0-Beta1);
	auto Vx=V/(1.0-Beta2);
	X = X + dt*Mx/(std::sqrt(Vx)+eps);
	this->raw(i)=X;
	Mt->raw(i)=M;
	Vt->raw(i)=V;
    };    
};
}; // class


#endif
