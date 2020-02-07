include ../makefiles/Makefile.header

cuda_sources := $(wildcard *.cu)
cpp_sources  := $(wildcard *.cpp)

cuda_objects := $(patsubst %.cu,%.o,$(cuda_sources))
cpp_objects  := $(patsubst %.cpp,%.o,$(cpp_sources))
clean_files  := $(cuda_objects) $(cpp_objects)
CPP_FLAGS    += -I$(shell pwd)/../include
NVCC_FLAGS   += -I$(shell pwd)/../include

%.o: %.cpp
	@$(CPP_COMPILER) $< $(CPP_FLAGS) -c -o $@

%.o: %.cu
	@$(COMPILER) $< $(NVCC_FLAGS) -c -o $@

default: $(cuda_objects) $(cpp_objects)
	@echo done >/dev/null

clean:
	@$(foreach var, $(clean_files), $(shell if [ -f $(var) ]; then rm $(var); fi;))
