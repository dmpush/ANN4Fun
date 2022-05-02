RM=rm -f
TARGET=test.GAN
all:
	clear
#	clang++ -std=c++20 -g -static -I .. -I ../../ -ljpeg main.cpp -o $(TARGET)
	g++ -Wall -Wextra -std=c++20 -g -I .. -I ../../ -I ../../units -ljpeg -lgomp -fopenmp main.cpp -o $(TARGET) -O2
#	g++ -std=c++20 -g -I .. -I ../../ -ljpeg  main.cpp -o $(TARGET)

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