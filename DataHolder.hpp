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

/**
    @brief DataHolder - массив памяти, хранящий в себе именованные переменные - тензоры. В некоторых случаях,
    объекты этого класса интерпретиуются как одномерные тензоры (вектор), например в реализации Учителя.
*/
template<typename T>
class DataHolder {
public:
    using sPtr=std::shared_ptr<DataHolder >;
    using uPtr=std::unique_ptr<DataHolder >;

    //-------------------------------------------------------------------
    /**
	@brief Tensor - класс, реализующий многомерный массив (тензор). Тензор не может
	существовать сам по себе, он всегда лежит в некотором хранилище, потому его реализция является
	внутренней по отношению к DataHolder.
    */
    class Tensor{
    public:
	friend class DataHolder;
        using sPtr=std::shared_ptr<Tensor>;

	Tensor() = delete;
        explicit Tensor(DataHolder *holder,  const std::vector<size_t>& dimensions) : holder_(holder) {
	    size_=1;
	    for(auto it : dimensions) {
		dims_.push_back(it);
		size_*=it;
	    };
	    offset_=0;
	};

        explicit Tensor(DataHolder *holder) : holder_(holder), size_{0}, dims_{}, offset_{0} {};

	Tensor(const Tensor&) = delete;
        virtual ~Tensor()=default;

	auto getHolder() { return holder_; };
	const size_t dim() { return dims_.size(); };
	const size_t size() { return size_; };
	const auto dims() { return dims_; };

	T& raw(size_t ind) { return holder_->raw(offset_+ind); };

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
	DataHolder *holder_;
	std::vector<size_t> dims_;
	size_t offset_;
	size_t size_;

	void setOffset(int offset)  { offset_=offset; };
    };
    //-------------------------------------------------------------------

    DataHolder(const DataHolder&) = delete;
    DataHolder() = default;
    virtual ~DataHolder() = default;
    /// модификация данных напрямую
    T& raw(size_t ind) { return data_[ind]; }; 
    /// возвращает указатель на тензор по его имени/ключу
    typename Tensor::sPtr get(std::string name)  {
	auto it=objects_.find(name);
	if(it==objects_.end())
	    throw std::runtime_error(std::string("Нет тензора ")+name+std::string(" в хранилище"));
	return objects_[name];
    };

    /// добавляет в хранилище пару имя тензора/форма тензора
    void append(std::string name, const std::vector<size_t>& dims) {
	auto obj=std::make_shared<Tensor>(this, dims);
	append(name, obj);
    };
    /// добавляет пустой тензор 
    void append(std::string name) {
	auto obj=std::make_shared<Tensor>(this);
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
    /// true, если хранилище пустое или неинициализированное командой build()
    bool isEmpty() { return size()==0; };
    std::random_device rdev_;
private:
    void append(std::string name, typename Tensor::sPtr obj) {
        objects_[name]=obj;
    };
    std::vector<T> data_;
    std::map<std::string, typename Tensor::sPtr> objects_;

}; //class
template <typename T>
    using Tensor=typename DataHolder<T>::Tensor::sPtr;
    
#endif