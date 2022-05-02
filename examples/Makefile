all:
	make -C units
	make -C MNIST.classifier
	make -C fuzzy2jpeg
	make -C MNIST.autoencoder
	make -C MNIST.GAN
run:
	make -C units run
clean:
	make -C units clean
	make -C MNIST.classifier clean
	make -C fuzzy2jpeg clean
	make -C MNIST.autoencoder clean
	make -C MNIST.GAN clean
