#ifndef __I_TENSOR_HPP__
#define __I_TENSOR_HPP__

#include <functional>
#include <vector>

template<typename T>
class IDataHolder;


/**
    @brief Tensor - класс, реализующий многомерный массив (тензор). 
*/
template<typename T>
class ITensor {
public:
    friend class IDataHolder<T>;
    using sPtr=std::shared_ptr<ITensor>;

    ITensor() = delete;
    explicit ITensor(IDataHolder<T> *holder,  const std::vector<size_t>& dimensions) : holder_(holder) {
	size_=1;
	for(auto it : dimensions) {
	    dims_.push_back(it);
	    size_*=it;
	};
	offset_=0;
    };

    explicit ITensor(IDataHolder<T> *holder) : holder_(holder), dims_{},  offset_{0}, size_{0} {};

    ITensor(const ITensor&) = delete;
    virtual ~ITensor()=default;

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
	if(std::size(ind)!=ITensor<T>::dim())
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
        assert(ITensor<T>::dim()==2);
        return raw(i+j*dims_[0]);
    };

    /// паттерн Прототип
    virtual typename ITensor<T>::sPtr clone() = 0;

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
protected:
    void setHolder(IDataHolder<T>* holder) { holder_=holder; };
    auto getHolder() { return holder_; };
    void setOffset(int offset)  { offset_=offset; };
    auto getOffset() { return offset_; };
private:
    IDataHolder<T> *holder_;
    std::vector<size_t> dims_;
    size_t offset_;
    size_t size_;
public:
/// @brief sum() - поэлементная сумма двух тензоров с записью в третий тензор: this=A+B
/// @param A,B - слагаемые
virtual void sum   (sPtr A, sPtr B) = 0;
/// @brief prod() - поэлементное произведение  двух тензоров с записью в третий тензор: this=A.*B
/// @param A,B - сомножители
virtual void prod   (sPtr A, sPtr B) = 0;
/// @brief prodapp() - поэлементное произведение  двух тензоров с суммированием с  третим тензором: this+=A.*B
/// @param A,B - сомножители
virtual void prodapp(sPtr A, sPtr B) = 0;
/// @brief append() - операция += для тензоров: this+=A
/// @param A -- аргумент
virtual void append   (sPtr A) = 0;
///  @brief mul() -  Умножение матриц, векторов и матриц, и т.д.: this=A*B
/// @param A,B -- сомножители
virtual void mul   (sPtr A, sPtr B) = 0;
/// @brief copy() -- название говорит само за себя. Копирование тензоров. this=src
/// @param src -- источник
virtual void copy  (sPtr src) = 0;
/// @brief fill() -- заполнение тензора постоянным значением val
/// @param val -- заполнитель тензора
virtual void fill  (T val) = 0;
/// @brief extmulapp() внешнее произведение двух векторов c добавлением к двухмерной матрице: this+=A*B
/// @param A,B -- сомножители
virtual void extmulapp(sPtr A, sPtr B) = 0;
/// @brief gaussianNoise - заполнение тензора шумом Гаусса
/// @param M -- математическое ожидание
/// @param S -- среднеквадатичное отклонение
virtual void gaussianNoise(T M, T S) = 0;
/// @brief uniformNoise - заполнение тензора шумом Гаусса
/// @param a -- нижняя граница значений шума
/// @param b -- верхняя граница значений шума
virtual void uniformNoise(double a, double b) = 0;
/// @brief распределение Бернулли
/// @param prob -- вероятнось выпадания орла (heads) либо решки (tails)
virtual void bernoulliNoise(double prob, T heads=1.0, T tails=0.0) = 0;
/// @brief apply -- применение функции ко всем элементам тензора
/// @param func -- функция
virtual void apply(const std::function<T(T)>&func) = 0;
}; // class

template<typename T>
    using TensorPtr=std::shared_ptr<ITensor<T>>;

#endif
