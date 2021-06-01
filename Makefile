RM=rm -f
all:
	clang++ -std=c++20 -static -g -I . main.cpp -o test
run:
	./test
clean:
	$(RM) test
	$(RM) callgrind.out*
leaks:
	valgrind --leak-check=full ./test
profile:
	valgrind --tool=callgrind ./test
view:
	qcachegrind 