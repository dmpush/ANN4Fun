RM=rm -f
TARGET=test
all:
	clear
	clang++ -std=c++20 -static -g -I . main.cpp -o $(TARGET)
run:
	./test
clean:
	make -C units clean
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