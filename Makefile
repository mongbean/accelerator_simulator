# Set SYSTEMC_HOME to your SystemC install (the directory containing include/ and lib-linux64/ or lib/)
SYSTEMC_HOME ?= /usr/local/systemc
CXXFLAGS ?= -std=c++17 -O2 -Wall -I$(SYSTEMC_HOME)/include
LDFLAGS  ?= -L$(SYSTEMC_HOME)/lib-linux64 -L$(SYSTEMC_HOME)/lib -Wl,-rpath,$(SYSTEMC_HOME)/lib-linux64 -Wl,-rpath,$(SYSTEMC_HOME)/lib
sim_v0: sim_v0.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lsystemc
clean:
	rm -f sim_v0
