#ifndef DATAHOLDER_HPP
#define DATAHOLDER_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <iterator> // std::size
#include <iostream>
#include <algorithm> //std::fill

template<typename T>
class DataHolder {
public:
    using sPtr=std::shared_ptr<DataHolder >;
    using wPtr=std::weak_ptr<DataHolder >;

    //-------------------------------------------------------------------
    class Tensor{
    public:
	friend class DataHolder;
        using wPtr=std::weak_ptr<Tensor>;
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

	Tensor(const Tensor&) = delete;
        virtual ~Tensor()=default;

	const size_t dim() { return dims_.size(); };
	const size_t size() { return size_; };
	const auto dims() { return dims_; };
	T get(int ind) { return holder_->get(offset_+ind); };
	T set(int ind, T val) { return (holder_->set(offset_+ind, val)); };

	T get(const std::vector<size_t>& ind) {
	    if(std::size(ind)!=dim())
		throw std::runtime_error("Неправильное число индексов в тензоре");
	    if(dim()==1)
		return get(ind[0]);
	    else if(dim()==2)
		return get(ind[0]+ind[1]*dims_[0]);
	    else if(dim()==3)
		return get(ind[0] + ind[1]*dims_[0] + ind[2]*dims_[0]*dims_[1]);
	    throw std::runtime_error("Тензор не реализован");
	    return T(0);
	};

	T set(const std::vector<size_t>& ind, T value) {
	    if(std::size(ind)!=dim())
		throw std::runtime_error("Неправильное число индексов в тензоре");
	    if(dim()==1)
		return set(ind[0], value);
	    else if(dim()==2)
		return set(ind[0]+ind[1]*dims_[0], value);
	    else if(dim()==3)
		return set(ind[0] + ind[1]*dims_[0] + ind[2]*dims_[0]*dims_[1], value);
	    throw std::runtime_error("Тензор не реализован");
	    return T(0);
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
//    friend void sum(typename Tensor::sPtr A, typename Tensor sPtr B, typename Tensor::sPtr res);

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

    T get(int ind) { return data_[ind]; };
    T set(int ind, T val) { return (data_[ind]=val); };
    typename Tensor::sPtr get(std::string name)  {
	return objects_[name];
    };


    void append(std::string name, std::initializer_list<size_t> dims) {
	auto obj=std::make_shared<Tensor>(this, dims);
	append(name, obj);
    };

    void build() {
	int offset=0;
	for(auto [name, obj]: objects_) {
	    obj->setOffset(offset);
	    offset+=obj->size();
	};
	data_.resize(offset);
	std::fill(data_.begin(), data_.end(), T(0));
    };
    size_t size() { return data_.size(); };
    /// оператор  A+=h*B
    void update(typename DataHolder<T>::sPtr B, T h) {
	if(size()!=B->size())
	    throw std::runtime_error("Градиент имеет неправильный размер");
	for(size_t i=0; i<size(); i++)
	data_[i] += h*B->data_[i];
    };
    /// шаблон Прототип
    auto clone() {
	auto out=std::make_shared<DataHolder<T>>();
	for(auto [name, obj] : objects_) {
	    auto o=obj->clone();
	    o->holder_=out.get();
	    out->append(name, o);
	};
	out->build();
	return out;
    };
    void fill(T val) {
        std::fill(data_.begin(), data_.end(), val);
    };
    void description() {
	std::cout<<"Размер хранища "<<size()<<" объектов ("<<size()*sizeof(T)<<" байт)."<<std::endl;
	for(auto [n,o]: objects_) {
	    std::cout<<n<<": ";
	    o->description();
	};
    };
    bool isEmpty() { return size()==0; };

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