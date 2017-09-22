# target
TARGET = rtpose.so 

# shell command
CXX = g++
NVCC = nvcc
RM = rm -fr

# custom DIR, modified according to your own
CAFFE_DIR = ./3rdparty/caffe

# directory
OUTDIR = build
SRCDIR = src
SUBDIRS = $(shell find $(SRCDIR) -type d)

INCDIR = -Isrc \
	-Isrc/include \
	-I/usr/include/python2.7 \
	-I/usr/local/cuda/include \
	-I$(CAFFE_DIR)/include \
	`pkg-config --cflags opencv`

LIBDIR = -L/usr/local/cuda/lib64 \
	-L$(CAFFE_DIR)/build/lib \
	`pkg-config --libs-only-L opencv`

LIBS = -lcudart \
	-lnppi \
	-lopencv_imgproc \
	-lopencv_highgui \
	-lopencv_core \
	-lcaffe \
    -lboost_thread \
    -lboost_date_time \
	-lboost_system \
	-lboost_filesystem \
    -lboost_chrono \
    -lboost_atomic \
    -lboost_python \
	-pthread \
	-lglog \
	-lgflags \
	-lpython2.7 \
      	`pkg-config --libs-only-l opencv`

	#-lopencv_imgcodecs \
	#-lopencv_videoio \
# cpp source file
SRCCPP = $(foreach dir,$(SUBDIRS),$(wildcard $(dir)/*.cpp))
CPPOBJS = $(foreach file, $(SRCCPP:.cpp=.cpp.o), $(OUTDIR)/$(file))
CPPDEP = $(foreach file, $(SRCCPP:.cpp=.cpp.d), $(OUTDIR)/$(file))

# cuda source file
SRCCU = $(foreach dir,$(SUBDIRS),$(wildcard $(dir)/*.cu))
CUOBJS = $(foreach file, $(SRCCU:.cu=.cu.o), $(OUTDIR)/$(file))
CUDEP = $(foreach file, $(SRCCU:.cu=.cu.d), $(OUTDIR)/$(file))

# object file
OBJS = $(CPPOBJS) $(CUOBJS)
DEPENDFILES = $(CPPDEP) $(CUDEP)

# Gencode arguments
SMS ?= 50 52 60 61

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

#CXXFLAGS = $(INCDIR) -g -fpermissive -std=c++11
#CXXFLAGS = $(INCDIR) -O3 -fpermissive -std=c++11
CXXFLAGS = $(INCDIR) -O3 -fPIC -fpermissive -std=c++11


NVCCFLAGS = $(GENCODE_FLAGS) $(INCDIR) -O3 --use_fast_math

LDFLAGS = $(LIBDIR) $(LIBS) -shared -Wl,-rpath=.:..:lib:$(CAFFE_DIR)/build/lib

#LDFLAGS = $(LIBDIR) $(LIBS) -Wl,-rpath=.:..:lib:$(CAFFE_DIR)/build/lib

# defination
.SUFFIXES: .cpp .h .d
.PHONY: mk_dir clean all echo

# rules
all: $(OUTDIR)/$(TARGET)

mk_dir:
	@[ -d $(OUTDIR) ] || mkdir -p $(OUTDIR); \
	for val in $(SUBDIRS); do \
		[ -d $(OUTDIR)/$${val} ] || mkdir -p  $(OUTDIR)/$${val};\
	done;

echo:
	@echo 'SUBDIRS:$(SUBDIRS)'
	@echo 'CXXFLAGS:$(CXXFLAGS)'
	@echo 'OBJS:$(OBJS)'
	@echo 'DEPENDFILES:$(DEPENDFILES)'
	@echo 'LDFLAGS:$(LDFLAGS)'

$(OUTDIR)/%.cpp.o:%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUTDIR)/%.cu.o:%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OUTDIR)/$(TARGET):$(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	-@ $(RM) $(OUTDIR)/* $(OUTDIR)/$(TARGET)

# source and header file dependent
-include $(DEPENDFILES)
$(OUTDIR)/%.cpp.d:%.cpp | mk_dir
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	$(RM) $@.$$$$
