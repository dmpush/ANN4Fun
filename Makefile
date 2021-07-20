RM=rm -f
TARGET=test
all:
	clear
#	g++ -std=c++20 -static -g -I . -lgomp -fopenmp main.cpp -o $(TARGET)
	clang++ -Wall -Wextra -std=c++20 -static -g -I . -I units main.cpp -o $(TARGET)
run:
	./test
clean:
	make -C units clean
	make -C examples clean
	$(RM) -r docs
	$(RM) $(TARGET)
	$(RM) callgrind.out*
leaks:
	valgrind --leak-check=full ./$(TARGET)
profile:
	valgrind --tool=callgrind ./$(TARGET)
view:
	qcachegrind 
unit:
	make -C units
	make -C examples
	make -C units run
	make -C examples run
doc:
	doxygen