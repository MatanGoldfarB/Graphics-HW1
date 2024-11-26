# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -Wall -I/opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4

# Linker flags
LDFLAGS = -L/opt/homebrew/Cellar/opencv/4.10.0_12/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# Target executable
TARGET = hw1

# Source files
SRCS = HW1.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)