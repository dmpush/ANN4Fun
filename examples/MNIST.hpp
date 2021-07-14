#ifndef __MNIST_HPP__
#define __MNIST_HPP__

#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
#include <memory>
#include <random>
#include <cassert>
#include <deque>
#include <DataHolder.hpp>

template <class T>
void endswap(T *objp)
{
	unsigned char *memp = reinterpret_cast<unsigned char *>(objp);
	std::reverse(memp, memp + sizeof(T));
}

template <typename T>
class MNIST
{
    std::random_device seed_;
    std::mt19937 rdev_;
public:
    class Image {
    public: 
	using Picture=std::array<T, 28u*28u>;
    private:
	Picture data_;
	size_t label_;
    public:
	Image() = delete;
	Image(typename Image::Picture&& pic, size_t&& lab) : data_{std::move(pic)}, label_{std::move(lab)} {};
	~Image() = default;
	using sPtr=std::shared_ptr<Image>;
	auto label() { return label_; };
	T raw(size_t index) { return data_[index]; };
	T val(size_t i, size_t j) { return data_[j*28u+i]; };
    };
    using sPtr = std::shared_ptr<MNIST>;
	class MNIST_set
	{
		MNIST *owner_;
	public:

		std::vector<std::shared_ptr<Image>> images_;

		MNIST_set(MNIST *owner, std::string folder = "../", std::string prefix = "train"): owner_(owner)
		{
			std::deque<typename Image::Picture> Limages;
			std::deque<size_t> Llabels;
			std::ifstream file(folder + prefix + "-images-idx3-ubyte");
			if (!file.is_open())
				std::cout << "file not found!" << std::endl;
			int32_t magic, numItems, picWidth, picHeight;
			file.read((char *)&magic, sizeof(int32_t));
			endswap(&magic);
			file.read((char *)&numItems, sizeof(int32_t));
			endswap(&numItems);
			file.read((char *)&picWidth, sizeof(int32_t));
			endswap(&picWidth);
			file.read((char *)&picHeight, sizeof(int32_t));
			endswap(&picHeight);
			assert(magic==2051);
			assert(numItems==60000 || numItems==10000);
			assert(picWidth==28);
			assert(picHeight==28);
			uint8_t img[28 * 28];
			for (int i = 0; i < numItems; i++){
				file.read((char *)img, 28 * 28);
				typename Image::Picture pic;
				for (size_t k = 0; k < 28 * 28; k++){
				    pic[k] = static_cast<T>(static_cast<float>(img[k]) / 255.0f);
				}
				Limages.emplace_back(pic);
			};
			file.close();

			file = std::ifstream(folder + prefix + "-labels-idx1-ubyte");
			if (!file.is_open())
				std::cout << "file not found!" << std::endl;
			file.read((char *)&magic, sizeof(int32_t));
			endswap(&magic);
			assert(magic==2049);
			file.read((char *)&numItems, sizeof(int32_t));
			endswap(&numItems);
			assert(numItems==60000 || numItems==10000);
			for (int i = 0; i < numItems; i++) {
				char label;
				file.read((char *)&label, 1);
				Llabels.emplace_back(static_cast<int>(label));
			};
			file.close();

		    assert(Limages.size()==Llabels.size());
		    std::cout<<Limages.size()<<std::endl;
		    images_.resize(Limages.size());
		    for(size_t i=0; i<Limages.size(); i++) {
			images_[i]=std::make_shared<Image>(std::move(Limages[i]), std::move(Llabels[i]));
		    };
		};
		typename Image::sPtr	getRandomSample() {
			std::uniform_int_distribution<int> dist(0, images_.size()-1);
			return images_[dist(owner_->rdev_)];
		};
		size_t numSamples() {
			return images_.size();
		}
		auto begin() { return images_.begin(); };
		auto end() { return images_.end(); };
                auto shuffle() {
	            std::deque<typename MNIST<T>::Image::sPtr> dataset;
		    std::vector<size_t> indexes(numSamples());
		    for(size_t n=0; n<numSamples(); n++)
			indexes[n]=n;
		    std::shuffle(indexes.begin(), indexes.end(), owner_->rdev_);
	            for(size_t n=0; n<numSamples(); n++)
	                dataset.push_back(images_[indexes[n]]);
                    return dataset;
                };
	}; // MNIST_set
private:	
    std::shared_ptr<MNIST_set> train_;
    std::shared_ptr<MNIST_set> test_;
public:

    MNIST(std::string path) : seed_{}, rdev_{seed_()} {
	train_ = std::make_shared<MNIST_set>(this, path, "train");
	test_  = std::make_shared<MNIST_set>(this, path, "t10k");
    };

    std::shared_ptr<MNIST_set> getTrainSet() { return train_; };
    std::shared_ptr<MNIST_set> getTestSet() { return test_; };

    void shuffle(std::deque<typename MNIST<T>::Image::sPtr>& dataset) {
	std::shuffle(dataset.begin(), dataset.end(), rdev_);
    };
};

#endif
