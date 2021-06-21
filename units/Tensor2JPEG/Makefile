RM=rm -f
TARGET=unit
all:
	clear
	g++ -m64 -Wall -Wextra -std=c++20 -g -I .. -I ../../ -ljpeg main.cpp -o $(TARGET).64
#	g++ -m32 -Wall -std=c++20 -g -I .. -I ../../ -ljpeg main.cpp -o $(TARGET).32

run:
#	./$(TARGET).32
	./$(TARGET).64
clean:
	$(RM) $(TARGET).32
	$(RM) $(TARGET).64
	$(RM) callgrind.out*
	$(RM) *.jpg
leaks:
	valgrind --leak-check=full ./$(TAGET).64
profile:
	valgrind --tool=callgrind ./$(TARGET).64
view:
	qcachegrind 