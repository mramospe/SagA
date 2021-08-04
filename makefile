HEADERS=$(shell find include -name "*.hpp")

test: test.cpp makefile $(HEADERS)
	g++ -pedantic -Wall -Wsuggest-override -Wshadow -Iinclude -std=c++17 -o test test.cpp
