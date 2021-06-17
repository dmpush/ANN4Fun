RM=rm -f
TARGET=unit
all:
	clear
	clang++ -std=c++20 -static -m32 -g -I .. -I ../.. main.cpp -o $(TARGET).32
	clang++ -std=c++20 -static -m64 -g -I .. -I ../.. main.cpp -o $(TARGET).64
run:
	make run32
	make run64
run32:
	./$(TARGET).32
run64:
	./$(TARGET).64
clean:
	$(RM) $(TARGET).*
	$(RM) callgrind.out*
leaks:
	valgrind --leak-check=full ./$(TAGET).3
profile:
	valgrind --tool=callgrind ./$(TARGET)
view:
	qcachegrind 