RM=rm -f
TARGET=unit
all:
	clear
#	clang++ -std=c++20 -g -static -I .. -I ../../ -ljpeg main.cpp -o $(TARGET)
	g++ -Wall -std=c++20 -g -I .. -I ../../ -ljpeg -lgomp -fopenmp main.cpp -o $(TARGET)
#	g++ -std=c++20 -g -I .. -I ../../ -ljpeg  main.cpp -o $(TARGET)

run:
	./$(TARGET)
clean:
	$(RM) $(TARGET)
	$(RM) callgrind.out*
	$(RM) *.jpg
leaks:
	valgrind --leak-check=full ./$(TAGET).3
profile:
	valgrind --tool=callgrind ./$(TARGET)
view:
	qcachegrind 