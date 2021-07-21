RM=rm -f
TARGET=unit
all:
#	clear
	g++ -Wall -Wextra -std=c++20 -g -I ../../units -I .. -I ../../ -ljpeg main.cpp -o $(TARGET) -lgomp -fopenmp

run:
	./$(TARGET)
clean:
	$(RM) $(TARGET)
	$(RM) callgrind.out*
	$(RM) *.jpg
leaks:
	valgrind --leak-check=full ./$(TARGET)
profile:
	valgrind --tool=callgrind ./$(TARGET)
view:
	qcachegrind 