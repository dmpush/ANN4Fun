RM=rm
all:
	g++ -std=c++20 -static -g -I . main.cpp -o test
run:
	./test
#clean:
#        $(RM) test
