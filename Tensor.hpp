#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <functional>
#include <vector>
//#include <DataHolder.hpp>

template<typename T>
class DataHolder;


/**
    @brief Tensor - класс, реализующий многомерный массив (тензор). 
*/
template<typename T>
class Tensor {
public:
    friend class DataHolder<T>;
    using sPtr=std::shared_ptr<Tensor>;

    Tensor() = delete;
    explicit Tensor(DataHolder<T> *holder,  const std::vector<size_t>& dimensions) : holder_(holder) {
	size_=1;
	for(auto it : dimensions) {
	    dims_.push_back(it);
	    size_*=it;
	};
	offset_=0;
    };

    explicit Tensor(DataHolder<T> *holder) : holder_(holder), dims_{},  offset_{0}, size_{0} {};

    Tensor(const Tensor&) = delete;
    virtual ~Tensor()=default;

    auto getHolder() { return holder_; };
    /// @brief dim -- возвращает размерность тензора
    /// @returns размерность тензора
    size_t dim()  const { return dims_.size(); };
    size_t size() const { return size_; };
    auto dims() const { return dims_; };
    /// @brief прямой доступ к элементу тензора по сквозному индексу
    /// @param ind - сквозной индекс
    /// @returns элемент тензора
    T& raw(size_t ind) { return holder_->raw(offset_+ind); };
    /// @brief обобщенный доступ к элементу тензора по индексам
    /// лучше не использовать без необходимости
    /// @param ind список инициализации -- массив индексов
    /// @returns элемент тензора
    T& val(const std::initializer_list<size_t>& ind) {
	if(std::size(ind)!=dim())
	    throw std::runtime_error("Неправильное число индексов в тензоре");
	size_t plane=1;
	size_t off=0;
	size_t k=0;
	for(auto i: ind) {
	    off+=i*plane;
	    plane *= dims_[k];
	    k++;
	};
	return raw(off);
    };
    /// @breif доступ к элементам двухмерного тензора
    /// @params i,j - индексы элемента тензора
    /// @returns элемент тензора
    T& val(size_t i, size_t j) {
        assert(dim()==2);
        return raw(i+j*dims_[0]);
    };
    /// паттерн Прототип
    auto clone() {
	auto out=std::make_shared<Tensor>(nullptr, dims_);
	out->setOffset(offset_);
	return out;
    };

    void description() {
	std::cout<<"<"<<offset_<<">: ";
	for(auto i: dims_)
	    std::cout<<i<<" ";
	std::cout<<std::endl;
    };

    void dump() {
	if(dim()==1) {
	    std::cout<<"{";
	    for(size_t k=0; k<size(); k++)
		std::cout<<raw(k)<<( k+1==size() ? "" : ", ");
	    std::cout<<"}"<<std::endl;
	}
	 else if(dim()==2) {
	    std::cout<<"{";
	    for(size_t q=0; q<dims_[1]; q++) {
		std::cout<<"{";
		for(size_t p=0; p<dims_[0]; p++)
		    std::cout<<val({p,q})<<( p+1==dims_[0] ? "" : ", ");
		std::cout<<"}"<<std::endl;
	    };
	    std::cout<<"}"<<std::endl;
	}
	else {
	    std::cout<<"Отображение тензора не релизованно."<<std::endl;
	};
    };

    void show() {
	std::cout<<"{";
	for(int i=0; i<size(); i++)
	    std::cout<<val(i)<<(i+1!=size() ? ", ": "}");
    };

private:
    DataHolder<T> *holder_;
    std::vector<size_t> dims_;
    size_t offset_;
    size_t size_;

    void setOffset(int offset)  { offset_=offset; };

public:
/// @brief sum() - поэлементная сумма двух тензоров с записью в третий тензор: this=A+B
/// @param A,B - слагаемые
void sum   (sPtr A, sPtr B) {
    if(dim()!=A->dim() || dim()!=B->dim())
	throw std::runtime_error("sum: Размерности тензоров различны");
    if(size()!=A->size() || size()!=B->size())
	throw std::runtime_error("sum: Размеры тензоров различны");
    for(size_t i=0; i<size(); i++)
	raw(i) = A->raw(i) + B->raw(i);
};
/// @brief prod() - поэлементное произведение  двух тензоров с записью в третий тензор: this=A.*B
/// @param A,B - сомножители
void prod   (sPtr A, sPtr B) {
    if(dim()!=A->dim() || dim()!=B->dim())
	throw std::runtime_error("sum: Размерности тензоров различны");
    if(size()!=A->size() || size()!=B->size())
	throw std::runtime_error("sum: Размеры тензоров различны");
    for(size_t i=0; i<size(); i++)
	raw(i) = A->raw(i) * B->raw(i);
};

/// @brief prodapp() - поэлементное произведение  двух тензоров с суммированием с  третим тензором: this+=A.*B
/// @param A,B - сомножители
void prodapp(sPtr A, sPtr B) {
    if(dim()!=A->dim() || dim()!=B->dim())
	throw std::runtime_error("sum: Размерности тензоров различны");
    if(size()!=A->size() || size()!=B->size())
	throw std::runtime_error("sum: Размеры тензоров различны");
    for(size_t i=0; i<size(); i++)
	raw(i) = raw(i) + A->raw(i) * B->raw(i);
};


/// @brief append() - операция += для тензоров: this+=A
/// @param A -- аргумент
void append   (sPtr A) {
    if(dim()!=A->dim() )
	throw std::runtime_error("append(): Размерности тензоров различны");
    if(size()!=A->size() )
	throw std::runtime_error("append(): Размеры тензоров различны");
    #pragma omp parallel for
    for(size_t i=0; i<size(); i++)
	raw(i) = raw(i) + A->raw(i);
};

///  @brief mul() -  Умножение матриц, векторов и матриц, и т.д.: this=A*B
/// @param A,B -- сомножители
void mul   (sPtr A, sPtr B) {
	    auto dimsA=A->dims();
	    auto dimsB=B->dims();

	    // матрицы разворачиваются строка за строкой
	    if(A->dim()==1 && B->dim()==1) {
		// скалярное произведение векторов
		if(A->size()!=B->size() || size()!=1)
		    throw std::runtime_error("Вектора не сцеплены");
		T s{0};
		for(size_t j=0; j<dimsB[1]; j++)
		    s+=A->raw(j)*B->raw(j);
		raw(0) = s;
	    } else if(A->dim()==1 && B->dim()==2) {
		// умножение вектора-столбца на матрицу
		if(A->size() != dimsB[1] || dimsB[0]!=size() )
		    throw std::runtime_error("Вектор и матрица не сцеплены");
		#pragma omp parallel for
		for(size_t i=0; i<dimsB[0]; i++) {
		    T s{0};
		    for(size_t j=0; j<dimsB[1]; j++)
			s+=A->raw(j)*B->val(i,j);
		    raw(i) = s;
		};
	    } else if(A->dim()==2 && B->dim()==1) {
		// умножение матрицы на вектор-столбец
		if(dimsA[0]!=B->size() || dimsA[1]!=size())
		    throw std::runtime_error("Матрица и вектор не сцеплены");
		#pragma omp parallel for
		for(size_t i=0; i<dimsA[1]; i++) {
		    T s{0};
		    for(size_t j=0; j<dimsA[0]; j++)
			s+=A->val(j,i) * B->raw(j);
		    raw(i) = s;
		};
	    } else {
		throw  std::runtime_error("Умножение не релизовано");
	    };
};

/// @brief copy() - название говорит само за себя. Копирование тензоров. this=src
/// @param src -- источник
void copy  (sPtr src) {
    if(src->size()!=size())
	throw  std::runtime_error("copy(): тензоры имеют разные размеры");
    #pragma omp parallel for
    for(size_t i=0; i<src->size(); i++)
	raw(i) = src->raw(i);
};

/// @brief extmulapp() внешнее произведение двух векторов c добавлением к двухмерной матрице: this+=A*B
/// @param A,B -- сомножители
void extmulapp(sPtr A, sPtr B) {
    if(A->dim()!=1 || B->dim()!=1 || dim()!=2) 
	throw std::runtime_error("extmul(): входные тензоры должны быть векторами, а выходной - матрицей");
    auto dimsA=A->dims();
    auto dimsB=B->dims();
    auto dimsC=dims();
    
    if(dimsA[0]!=dimsC[0] || dimsB[0] != dimsC[1])
	throw std::runtime_error("extmulapp(): входные тензоны не согласованны с выходным");
    #pragma omp parallel for
    for(size_t i=0; i<dimsC[0]; i++)
	for(size_t j=0; j<dimsC[1]; j++)
	    val({i,j}) = val({i,j}) + A->raw(i) * B->raw(j);
};

/// @brief gaussianNoise - заполнение тензора шумом Гаусса
/// @param M -- математическое ожидание
/// @param S -- среднеквадатичное отклонение
void gaussianNoise(T M, T S) {
    auto holder=getHolder();
    for(size_t i=0; i<size(); i++)
	raw(i) = holder->gaussianNoise()*S + M; ;
    };

/// @brief uniformNoise - заполнение тензора шумом Гаусса
/// @param a -- нижняя граница значений шума
/// @param b -- верхняя граница значений шума
void uniformNoise(double a, double b) {
    auto holder=getHolder();
    for(size_t i=0; i<size(); i++)
	raw(i) = holder->uniformNoise() * (b-a)+a;
    };
/// @brief распределение Бернулли
/// @param prob -- вероятнось выпадания орла (heads) либо решки (tails)
void bernoulliNoise(double prob, T heads=1.0, T tails=0.0) {
    auto holder=getHolder();
    for(size_t i=0; i<size(); i++)
	raw(i) = holder->uniformNoise() < prob ? heads : tails;
    };
/// @brief apply -- применение функции ко всем элементам тензора
/// @param func -- функция
void apply(const std::function<T(T)>&func) {
    for(size_t i=0; i<size(); i++)
	raw(i) = func(raw(i));
};

}; // class

template<typename T>
    using TensorPtr=std::shared_ptr<Tensor<T>>;

#endif
