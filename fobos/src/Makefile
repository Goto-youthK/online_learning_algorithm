PROGRAM := main
SOURCES := $(wildcard *.cpp)
OBJS := $(SOURCES:.cpp=.o)
DEPENDS := $(SOURCES:.cpp=.depend~)
CC := gcc
CXX := g++
CPPFLAGS := -Wall -O2
LIBS := -lm

.PHONY: all
all: $(PROGRAM)
$(PROGRAM): $(OBJS)
	$(CXX) -o $(PROGRAM) $(CPPFLAGS) $(LIBS) $^

.PHONY: $(DEPENDS)
%.depend~: %.cpp
	@echo generating $@
	@$(SHELL) -ec '$(CC) -MM $(CPPFLAGS) $< | sed "s/\($*\)\.o[ :]*/\1.o $@ : /g" > $@; [ -s $@ ] || rm -f $@'

ifneq "$(MAKECMDGOALS)" "clean"
-include $(DEPENDS)
endif

.PHONY : clean
clean:
	rm -f $(PROGRAM)
	rm -f $(OBJS)
	rm -f *~
