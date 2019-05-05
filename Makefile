PROJ = convolve
CC = g++
NVCC = nvcc

CFLAGS = -c -g -Wall -I/opt/local/include -I$(HOME)/cppunit/include
CFLAGS += $(shell pkg-config --cflags $(OPENCV))
LDFLAGS = -L/opt/local/lib -L$(HOME)/cppunit/lib
LIBS = -lcppunit -ldl
LIBS += $(shell pkg-config --libs $(OPENCV))
CU_SRC = $(shell find ./ -name *.cu)
CPP_SRC = $(shell find ./ -name *.cpp)
OBJS = $(CPP_SRC:%.cpp=%.o) $(CU_SRC:%.cu=%.o)

INC_DIRS = ./
INC_FLAGS = $(addprefix -I, $(INC_DIRS))

all: $(PROJ)
	echo $(OBJS)

$(PROJ): $(OBJS) $(TEST_CASES)
	$(NVCC) -arch=sm_61 $(LDFLAGS) $(INC_FLAGS) $^ -o $@ $(LIBS)

%.o : %.cu
	$(NVCC) -arch=sm_61 -c $< -o $@

%.o : %.cpp
	$(CC) $(CFLAGS) $(INC_FLAGS) $< -o $@ 

clean:
	rm -f $(PROJ) $(OBJS) $(TEST_CASES) *.xml
