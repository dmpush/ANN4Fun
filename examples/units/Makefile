RM=rm -f
TARGET=unit
all:
#	clear
	clang++ -Wall -Wextra -std=c++20 -g -I .. -I ../../ -ljpeg main.cpp -o $(TARGET)

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