RM=rm -f
TARGET=test
all:
	clear
	clang++ -std=c++20 -static -g -I . main.cpp -o $(TARGET)
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
	make -C units run
	make -C examples
	make -C examples run
doc:
	doxygen